"""Runtime calibration and deterministic unit inference for DB memory."""

from __future__ import annotations

from typing import Any

from .config import db_memory_options_from_runtime_metadata
from .contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    extract_db_memory_semantic_contract,
)
from .records import DBMemoryRecord
from .storage import has_db_memory_marker


async def calibrate_db_memory(
    runtime: Any,
    *,
    source_owner: str,
    marker_key: str,
) -> dict[str, Any]:
    """Infer simple DB unit conventions and persist them as DB memory records."""
    try:
        memory_plugin = runtime.registry.get_plugin("memory")
    except KeyError:
        return {"calibrated": False, "reason": "memory_not_registered"}

    memory_options = db_memory_options_from_runtime_metadata(
        getattr(runtime.config, "metadata", {})
    )
    if await has_db_memory_marker(
        memory_plugin,
        marker_key,
        source_identity=str(memory_options.get("source_identity") or ""),
    ):
        return {"calibrated": False, "reason": "marker_exists"}

    schema_evidence = await runtime.execute_capability(
        "db.schema.inspect",
        owner=source_owner,
        operation_type="source.profile",
        input={},
    )
    schema = schema_evidence[0].payload if schema_evidence else {}
    records = unit_records_from_schema(schema)
    results = [
        await _write_db_memory_record_runtime(runtime, record) for record in records
    ]
    marker = await _write_db_memory_record_runtime(
        runtime,
        DBMemoryRecord(
            kind="cache_marker",
            key=marker_key,
            text=_marker_content(marker_key),
            importance=0.1,
            metadata={"exact": True},
        ),
    )
    return {
        "calibrated": True,
        "record_count": len(records),
        "records": results,
        "marker": marker,
    }


async def _write_db_memory_record_runtime(
    runtime: Any, record: DBMemoryRecord
) -> dict[str, Any]:
    """Write one DB memory record through the runtime capability boundary."""
    record = _record_with_runtime_source_identity(runtime, record)
    evidence = await runtime.execute_capability(
        "memory.semantic.write",
        owner="memory",
        operation_type="memory.update",
        input={
            "db_memory_payload": record.to_dict(),
            "db_memory_prompt": record.text,
        },
    )
    if not evidence:
        return {"success": False, "error": "memory write produced no evidence"}
    return dict(evidence[0].payload)


def _record_with_runtime_source_identity(
    runtime: Any,
    record: DBMemoryRecord,
) -> DBMemoryRecord:
    memory_options = db_memory_options_from_runtime_metadata(
        getattr(runtime.config, "metadata", {})
    )
    source_identity = memory_options.get("source_identity")
    metadata = dict(record.metadata)
    if source_identity:
        metadata.setdefault("source_identity", source_identity)
    metadata.setdefault("workspace_scope", "source")
    metadata.setdefault("active", True)
    metadata.setdefault("creation_path", "runtime_calibration")
    return DBMemoryRecord(
        kind=record.kind,
        key=record.key,
        text=record.text,
        metadata=metadata,
        importance=record.importance,
    )


def unit_records_from_schema(
    schema: dict[str, Any],
) -> tuple[DBMemoryRecord, ...]:
    """Infer obvious numeric unit conventions from schema metadata."""
    records: list[DBMemoryRecord] = []
    for column in _numeric_columns(schema):
        unit, reason = _infer_unit_from_column_name(column["column"])
        if unit is None:
            continue
        confidence = "high" if unit in {"cents", "percent", "basis_points"} else "low"
        table_name = column["table"]
        column_name = column["column"]
        metadata = {
            "table": table_name,
            "column": column_name,
            "unit": unit,
            "confidence": confidence,
            "reason": reason,
        }
        draft = DBMemoryRecord(
            kind="unit_convention",
            key=f"unit_convention:{table_name}.{column_name}",
            text=(
                f"{table_name}.{column_name} is stored as {unit} "
                f"(confidence: {confidence}). Reason: {reason}"
            ),
            metadata=metadata,
            importance=0.75 if confidence == "high" else 0.65,
        )
        contract = extract_db_memory_semantic_contract(draft, schema=schema)
        if contract is not None:
            metadata[DB_MEMORY_SEMANTIC_CONTRACT_KEY] = contract
            metadata["semantic_contract_status"] = "validated"
        records.append(
            DBMemoryRecord(
                kind=draft.kind,
                key=draft.key,
                text=draft.text,
                metadata=metadata,
                importance=draft.importance,
            )
        )
    return tuple(records)


def _numeric_columns(schema: dict[str, Any]) -> list[dict[str, Any]]:
    numeric = []
    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        for column in table.get("columns", []) or []:
            column_name = str(column.get("name") or "").strip()
            column_type = str(
                column.get("data_type") or column.get("type") or ""
            ).strip()
            if column_name and _is_numeric_type(column_type):
                entry: dict[str, Any] = {
                    "table": table_name,
                    "column": column_name,
                    "type": column_type,
                }
                if column.get("_samples"):
                    entry["samples"] = column["_samples"]
                if column.get("column_comment"):
                    entry["comment"] = column["column_comment"]
                numeric.append(entry)
    return numeric


def _is_numeric_type(value: str) -> bool:
    text = value.lower()
    return any(
        token in text
        for token in (
            "bigint",
            "decimal",
            "double",
            "float",
            "int",
            "numeric",
            "number",
            "real",
        )
    )


def _infer_unit_from_column_name(column: str) -> tuple[str | None, str | None]:
    text = column.lower()
    if "cents" in text or text.endswith("_cent"):
        return "cents", "column name contains cents"
    if "basis_points" in text or text.endswith("_bps") or text == "bps":
        return "basis_points", "column name indicates basis points"
    if "percent" in text or text.endswith("_pct") or text.endswith("_percentage"):
        return "percent", "column name indicates percent"
    return None, None


def _marker_content(key: str) -> str:
    return f"DB exact cache marker: {key}"
