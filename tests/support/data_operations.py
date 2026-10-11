"""Fixed analytical cases and independent oracles for live data operations.

These are fixture facts, not instructions, runtime permissions or an alternate
execution path. SQL under evaluation never supplies the expected answer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from daita._json import canonical_json

AUTHORIZATION = "DAITA_RUN_LIVE_DATA_OPERATIONS"
REPEATS_ENV = "DAITA_DATAOPS_REPEATS"
REPORT_ENV = "DAITA_DATAOPS_REPORT_DIR"
TOTAL_COST_ENV = "DAITA_DATAOPS_TOTAL_COST_USD"

KINDS = ("paid_totals", "refund_totals", "join_grain", "local_day", "paid_count")
METRIC_DEFINITION = (
    "Our settlement reserve means 3 percent of orders.amount_minor, rounded to "
    "whole minor currency units per order with Decimal ROUND_HALF_UP. Eligible "
    "orders have status='paid'. Settlement contribution is amount_minor minus "
    "refund_minor minus that reserve. Group results by currency and never combine "
    "currencies. This definition applies only to the admitted orders table."
)
METRIC_QUESTION = (
    "Calculate our settlement contribution by currency from orders. Use the "
    "approved business definition; if it is unavailable, say so instead of "
    'inventing it. Return only a JSON object shaped as {"rows":'
    '[{"currency":"USD","amount_minor":123}]} (never a bare array), or '
    'an object shaped as {"clarification":"..."} when the definition is unavailable.'
)


@dataclass(frozen=True)
class AnalyticalCase:
    kind: str
    variant: int

    @property
    def case_id(self) -> str:
        return f"{self.kind}-{self.variant:02}"

    @property
    def prompt(self) -> str:
        questions = {
            "paid_totals": (
                "Sum amount_minor minus refund_minor for status='paid' orders, "
                "separately by currency."
            ),
            "refund_totals": (
                "Sum refund_minor across all orders, separately by currency. "
                "Include zero totals for currencies present in orders."
            ),
            "join_grain": (
                "For paid orders having at least one item, sum the order's "
                "amount_minor exactly once per order, separately by currency. "
                "The existence of several order_items must not multiply amounts."
            ),
            "local_day": (
                "Sum amount_minor of paid orders whose created_at falls on "
                "2026-09-06 in America/Chicago, separately by currency. Convert "
                "the UTC timestamps to that timezone before filtering."
            ),
            "paid_count": (
                "Count distinct paid order_id values, separately by currency. "
                "Return that count in amount_minor for this count-only fixture."
            ),
        }
        return (
            questions[self.kind]
            + " Query current admitted data; do not change data or save memory. "
            'Return only JSON in the shape {"rows":[{"currency":"USD",'
            '"amount_minor":123}]}. Include exactly the groups requested.'
        )


CASES = tuple(AnalyticalCase(kind, variant) for kind in KINDS for variant in range(10))


def repetitions() -> range:
    value = int(os.environ.get(REPEATS_ENV, "1"))
    if not 1 <= value <= 5:
        raise ValueError(f"{REPEATS_ENV} must be between one and five")
    return range(value)


def require_authorization() -> None:
    if os.environ.get(AUTHORIZATION) != "1":
        raise ValueError(f"{AUTHORIZATION}=1 is required before live construction")


def fixture_rows(variant: int = 0) -> tuple[dict[str, Any], ...]:
    if not 0 <= variant < 10:
        raise ValueError("fixture variant must be between zero and nine")
    specifications = (
        ("USD", 10001, 101, "paid", "2026-09-06T04:59:59+00:00"),
        ("USD", 20005, 0, "paid", "2026-09-06T05:00:00+00:00"),
        ("USD", 90000, 0, "pending", "2026-09-06T12:00:00+00:00"),
        ("EUR", 12050, 50, "paid", "2026-09-07T04:59:59+00:00"),
        ("EUR", 19999, 19999, "refunded", "2026-09-06T12:00:00+00:00"),
        ("EUR", 3333, 0, "paid", "2026-09-07T05:00:00+00:00"),
        ("USD", 7777, 77, "paid", "2026-09-06T15:00:00+00:00"),
        ("EUR", 2501, 1, "paid", "2026-09-06T08:00:00+00:00"),
    )
    return tuple(
        {
            "order_id": index,
            "currency": currency,
            "amount_minor": amount + variant * (index + 11),
            "refund_minor": refund,
            "status": status,
            "created_at": created,
        }
        for index, (currency, amount, refund, status, created) in enumerate(
            specifications, 1
        )
    )


def item_rows() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (order_id * 10 + item, order_id, item)
        for order_id, count in ((1, 3), (2, 2), (3, 1), (4, 4), (6, 1), (8, 2))
        for item in range(1, count + 1)
    )


def create_database(path: Path, variant: int = 0) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY, currency TEXT NOT NULL, "
            "amount_minor INTEGER NOT NULL, refund_minor INTEGER NOT NULL, "
            "status TEXT NOT NULL, created_at TEXT NOT NULL);"
            "CREATE TABLE order_items(item_id INTEGER PRIMARY KEY, "
            "order_id INTEGER NOT NULL REFERENCES orders(order_id), quantity INTEGER NOT NULL);"
        )
        connection.executemany(
            "INSERT INTO orders VALUES(:order_id,:currency,:amount_minor,:refund_minor,"
            ":status,:created_at)",
            fixture_rows(variant),
        )
        connection.executemany("INSERT INTO order_items VALUES(?,?,?)", item_rows())


def fixture_digest(variant: int = 0) -> str:
    return sha256(
        canonical_json({"orders": fixture_rows(variant), "items": item_rows()}).encode()
    ).hexdigest()


def expected_rows(kind: str, variant: int = 0) -> list[dict[str, object]]:
    from decimal import ROUND_HALF_UP

    totals: dict[str, int] = defaultdict(int)
    orders_with_items = {row[1] for row in item_rows()}
    for row in fixture_rows(variant):
        currency = row["currency"]
        if kind == "refund_totals":
            totals[currency] += row["refund_minor"]
            continue
        if row["status"] != "paid":
            continue
        if kind == "join_grain" and row["order_id"] not in orders_with_items:
            continue
        if kind == "local_day":
            instant = datetime.fromisoformat(row["created_at"])
            if (
                instant.astimezone(ZoneInfo("America/Chicago")).date().isoformat()
                != "2026-09-06"
            ):
                continue
        if kind == "paid_count":
            value = 1
        elif kind == "settlement_contribution":
            reserve = int(
                (Decimal(row["amount_minor"]) * Decimal("0.03")).quantize(
                    Decimal(1), rounding=ROUND_HALF_UP
                )
            )
            value = row["amount_minor"] - row["refund_minor"] - reserve
        elif kind == "paid_totals":
            value = row["amount_minor"] - row["refund_minor"]
        elif kind in {"join_grain", "local_day"}:
            value = row["amount_minor"]
        else:
            raise ValueError(f"unknown oracle kind: {kind}")
        totals[currency] += value
    return [
        {"currency": currency, "amount_minor": amount}
        for currency, amount in sorted(totals.items())
    ]


def parse_answer(text: str | None) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = (text or "").strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    data = json.loads(value, object_pairs_hook=unique_pairs, parse_float=Decimal)
    if not isinstance(data, dict):
        raise ValueError("answer must be one JSON object")
    return data


def score_answer(
    text: str | None, expected: list[dict[str, object]]
) -> dict[str, object]:
    try:
        answer = parse_answer(text)
        rows = answer["rows"]
        if set(answer) != {"rows"} or not isinstance(rows, list):
            raise ValueError("answer must contain exactly rows")
        actual = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"currency", "amount_minor"}:
                raise ValueError("row fields differ from the result contract")
            amount = row["amount_minor"]
            if isinstance(amount, bool) or not isinstance(amount, (int, Decimal)):
                raise ValueError("minor-unit amounts must be numeric")
            number = Decimal(amount)
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("minor-unit amounts must be finite integers")
            if not isinstance(row["currency"], str):
                raise ValueError("currency must be text")
            actual.append({"currency": row["currency"], "amount_minor": int(number)})
        actual.sort(key=lambda row: str(row["currency"]))
        return {"correct": actual == expected, "expected": expected, "actual": actual}
    except (ValueError, KeyError, TypeError):
        return {"correct": False, "expected": expected, "actual": None}
