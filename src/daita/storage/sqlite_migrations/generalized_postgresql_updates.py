"""Normalize durable records required by generalized PostgreSQL updates."""

from __future__ import annotations

import sqlite3

from ...llm.pricing import CostEstimateStatus
from ..sqlite_codecs import decode_loop_exit, decode_receipt
from ..sqlite_codecs.common import (
    decimal_decode,
    dump_payload,
    enum_encode,
    load_payload,
    record,
    record_fields,
)
from ..sqlite_schema import GENERALIZED_UPDATE_TABLES, SCOPED_PERMISSION_TABLES
from .models import SQLiteMigration
from .scoped_source_permissions import validate_target as validate_scopes

MIGRATION_ID = "20260814_generalized_postgresql_updates"
DEFINITION = """20260814_generalized_postgresql_updates
add expected_affected_rows equal to one to every historical one-row database write receipt;
normalize historical scalar run-cost estimates without claiming complete pricing;
retain the existing scoped-permission physical schema and every durable identity
"""

_LEGACY_RECEIPT_FIELDS = (
    "receipt_id",
    "agent_id",
    "run_id",
    "call_id",
    "capability_id",
    "source_id",
    "resource_id",
    "intent_sha256",
    "preview_fingerprint",
    "outcome",
    "affected_rows",
    "normalized_error_code",
    "started_at",
    "completed_at",
)
_LEGACY_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "estimated_cost_usd",
}
_CURRENT_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_estimate",
}


def _normalize_run_results(connection: sqlite3.Connection) -> None:
    for run_id, conversation_id, data in tuple(
        connection.execute(
            "SELECT id, conversation_id, result FROM runs WHERE result IS NOT NULL"
        )
    ):
        payload = load_payload(data)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"__record__", "fields"}
            or payload["__record__"] != "LoopExit"
            or not isinstance(payload["fields"], dict)
        ):
            raise ValueError("stored LoopExit record envelope is invalid")
        usage = payload["fields"].get("usage")
        if (
            not isinstance(usage, dict)
            or set(usage) != {"__record__", "fields"}
            or usage["__record__"] != "ModelUsage"
            or not isinstance(usage["fields"], dict)
        ):
            raise ValueError("stored ModelUsage record envelope is invalid")
        usage_fields = usage["fields"]
        if set(usage_fields) == _LEGACY_USAGE_FIELDS:
            historical_amount = usage_fields.pop("estimated_cost_usd")
            decimal_decode(historical_amount)
            usage_fields["cost_estimate"] = record(
                "CostEstimate",
                {
                    "amount_usd": historical_amount,
                    "status": enum_encode(
                        CostEstimateStatus.PARTIAL,
                        "CostEstimateStatus",
                    ),
                    "basis": None,
                    "rate_schedule_id": None,
                    "components": [],
                    "code": "legacy_estimate_completeness_unknown",
                },
            )
            encoded = dump_payload(payload)
            exit_record = decode_loop_exit(encoded)
            if (
                exit_record.run_id != run_id
                or exit_record.conversation_id != conversation_id
            ):
                raise ValueError("stored run result ownership is invalid")
            result = connection.execute(
                "UPDATE runs SET result = ? WHERE id = ? AND result = ?",
                (encoded, run_id, data),
            )
            if result.rowcount != 1:
                raise RuntimeError("stored run result changed during migration")
        elif set(usage_fields) == _CURRENT_USAGE_FIELDS:
            exit_record = decode_loop_exit(data)
            if (
                exit_record.run_id != run_id
                or exit_record.conversation_id != conversation_id
            ):
                raise ValueError("stored run result ownership is invalid")


def apply(connection: sqlite3.Connection) -> None:
    _normalize_run_results(connection)
    for agent_id, receipt_id, run_id, call_id, data in tuple(
        connection.execute("""SELECT agent_id, id, run_id, call_id, data
               FROM database_write_receipts""")
    ):
        fields = record_fields(
            load_payload(data),
            "DatabaseWriteReceipt",
            _LEGACY_RECEIPT_FIELDS,
        )
        fields["expected_affected_rows"] = 1
        encoded = dump_payload(record("DatabaseWriteReceipt", fields))
        receipt = decode_receipt(encoded)
        if (
            receipt.agent_id != agent_id
            or receipt.receipt_id != receipt_id
            or receipt.run_id != run_id
            or receipt.call_id != call_id
        ):
            raise ValueError("stored database write receipt ownership is invalid")
        result = connection.execute(
            """UPDATE database_write_receipts SET data = ?
               WHERE agent_id = ? AND id = ? AND data = ?""",
            (encoded, agent_id, receipt_id, data),
        )
        if result.rowcount != 1:
            raise RuntimeError("database write receipt changed during migration")


def validate_target(connection: sqlite3.Connection) -> None:
    validate_scopes(connection)
    for agent_id, receipt_id, run_id, call_id, data in connection.execute(
        """SELECT agent_id, id, run_id, call_id, data
           FROM database_write_receipts"""
    ):
        receipt = decode_receipt(data)
        if (
            receipt.agent_id != agent_id
            or receipt.receipt_id != receipt_id
            or receipt.run_id != run_id
            or receipt.call_id != call_id
        ):
            raise ValueError("stored database write receipt ownership is invalid")


MIGRATION = SQLiteMigration(
    ordinal=4,
    migration_id=MIGRATION_ID,
    definition=DEFINITION,
    source_schema=SCOPED_PERMISSION_TABLES,
    target_schema=GENERALIZED_UPDATE_TABLES,
    apply=apply,
    validate_target=validate_target,
)
