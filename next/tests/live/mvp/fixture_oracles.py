"""Deterministic Wave 1 commerce fixture and hard-oracle calculations.

The live model never supplies an expected value.  Every expected aggregate,
resource choice, and discrepancy is recomputed from the test-owned SQLite and
local-file fixture created from ``fixture_manifest.json``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import cast

from daita.domains.data import (
    TABULAR_COMPARISON_POLICY_SCHEMA_VERSION,
    TabularComparisonResult,
    TabularEvidenceDataset,
    compare_tabular_datasets,
)

MANIFEST_PATH = Path(__file__).with_name("fixture_manifest.json")
EXPECTED_FIXTURE_VERSION = "wave1-commerce-v1"
# This is an intentional review anchor over the canonical JSON content.  The
# fixture self-test fails until a manifest edit updates this digest explicitly.
EXPECTED_MANIFEST_SHA256 = (
    "sha256:a313d61d1e4dc0411b02e3d3314ba49b18dc238ea62621f1d77b9f5b4f01439c"
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
CROSS_SOURCE_KEY_NORMALIZATION = "stringify_integral"
_CUSTOMER_COMPARE_COLUMNS = (
    "id",
    "name",
    "plan",
    "lifecycle_status",
    "email",
)
_CUSTOMER_VALUE_COLUMNS = _CUSTOMER_COMPARE_COLUMNS[1:]
_PRODUCTION_DISCREPANCY_KINDS = {
    "duplicate_key",
    "invalid_key",
    "left_only",
    "missing_value",
    "right_only",
    "type_mismatch",
    "value_mismatch",
}


@dataclass(frozen=True, slots=True)
class CommerceFixture:
    root: Path
    database_path: Path
    files_root: Path
    fixture_version: str
    manifest_digest: str
    reporting_start: str
    reporting_end_exclusive: str


@dataclass(frozen=True, slots=True)
class Discrepancy:
    kind: str
    customer_id: str | None
    column: str | None = None
    file_value: object | None = None
    database_value: object | None = None
    file_present: bool | None = None
    database_present: bool | None = None
    file_type: str | None = None
    database_type: str | None = None
    source: str | None = None
    missing_columns: tuple[str, ...] = ()
    null_columns: tuple[str, ...] = ()
    duplicate_count: int | None = None


@dataclass(frozen=True, slots=True)
class AggregateOracle:
    customer_count: int
    net_revenue_cents: int

    @property
    def net_revenue(self) -> Decimal:
        return Decimal(self.net_revenue_cents) / Decimal(100)


@dataclass(frozen=True, slots=True)
class RegionOracle:
    region_name: str
    net_revenue_cents: int

    @property
    def net_revenue(self) -> Decimal:
        return Decimal(self.net_revenue_cents) / Decimal(100)


@dataclass(frozen=True, slots=True)
class CommerceOracles:
    aggregate: AggregateOracle
    leading_region: RegionOracle
    plan_breakdown: Mapping[str, AggregateOracle]
    enterprise_refunded_customer_count: int
    enterprise_refunded_cents: int
    newest_customer_export: Path
    comparison_key_normalization: str
    comparison_policy_schema_version: int
    discrepancies: tuple[Discrepancy, ...]


def load_manifest() -> dict[str, object]:
    decoded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("fixture manifest must contain one JSON object")
    return cast(dict[str, object], decoded)


def canonical_manifest_bytes(manifest: Mapping[str, object] | None = None) -> bytes:
    value = load_manifest() if manifest is None else manifest
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def manifest_digest(manifest: Mapping[str, object] | None = None) -> str:
    return "sha256:" + hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def build_commerce_fixture(root: Path) -> CommerceFixture:
    manifest = load_manifest()
    fixture_version = _text(manifest, "fixture_version")
    if fixture_version != EXPECTED_FIXTURE_VERSION:
        raise ValueError("fixture manifest version does not match the harness")
    reporting = _mapping(manifest, "reporting_period")
    reporting_start = _text(reporting, "start_inclusive")
    reporting_end = _text(reporting, "end_exclusive")
    database = _mapping(manifest, "database")

    root.mkdir(parents=True, exist_ok=False)
    database_path = root / _safe_filename(_text(database, "filename"))
    _create_database(database_path, database)

    files_root = root / "exports"
    files_root.mkdir()
    files = _sequence(manifest, "files")
    for raw_file in files:
        if not isinstance(raw_file, Mapping):
            raise ValueError("fixture file entry must be an object")
        _create_file(files_root, cast(Mapping[str, object], raw_file))

    return CommerceFixture(
        root=root,
        database_path=database_path,
        files_root=files_root,
        fixture_version=fixture_version,
        manifest_digest=manifest_digest(manifest),
        reporting_start=reporting_start,
        reporting_end_exclusive=reporting_end,
    )


def compute_oracles(fixture: CommerceFixture) -> CommerceOracles:
    aggregate = _aggregate_oracle(fixture)
    leading_region = _leading_region_oracle(fixture)
    plan_breakdown = _plan_breakdown_oracle(fixture)
    refunded_customers, refunded_cents = _enterprise_refund_oracle(fixture)
    newest = _newest_customer_export(fixture)
    comparison = _comparison_oracle(fixture, newest)
    return CommerceOracles(
        aggregate=aggregate,
        leading_region=leading_region,
        plan_breakdown=plan_breakdown,
        enterprise_refunded_customer_count=refunded_customers,
        enterprise_refunded_cents=refunded_cents,
        newest_customer_export=newest,
        comparison_key_normalization=comparison[1],
        comparison_policy_schema_version=comparison[2],
        discrepancies=comparison[0],
    )


def money_text(cents: int) -> str:
    return f"${Decimal(cents) / Decimal(100):,.2f}"


def _create_database(path: Path, database: Mapping[str, object]) -> None:
    ddl = _sequence(database, "ddl")
    rows = _mapping(database, "rows")
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in ddl:
            if not isinstance(statement, str) or not statement.startswith(
                "CREATE TABLE "
            ):
                raise ValueError("fixture DDL must contain CREATE TABLE statements")
            connection.execute(statement)
        for table, raw_rows in rows.items():
            if not isinstance(table, str) or _IDENTIFIER.fullmatch(table) is None:
                raise ValueError("fixture row table uses an unsafe identifier")
            if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
                raise ValueError("fixture table rows must be a sequence")
            columns = tuple(
                cast(str, row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            )
            placeholders = ", ".join("?" for _ in columns)
            column_sql = ", ".join(f'"{column}"' for column in columns)
            insert_sql = f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})'
            connection.executemany(insert_sql, tuple(tuple(row) for row in raw_rows))


def _create_file(root: Path, entry: Mapping[str, object]) -> None:
    filename = _safe_filename(_text(entry, "filename"))
    path = root / filename
    rows = _sequence(entry, "rows")
    if filename.endswith(".csv"):
        columns = _sequence(entry, "columns")
        if any(not isinstance(column, str) for column in columns):
            raise ValueError("fixture CSV columns must be strings")
        if any(
            not isinstance(row, Sequence) or isinstance(row, (str, bytes))
            for row in rows
        ):
            raise ValueError("fixture CSV rows must be arrays")
        csv_rows = cast(Sequence[Sequence[object]], rows)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(csv_rows)
    elif filename.endswith(".json"):
        path.write_text(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    else:
        raise ValueError("fixture files must use CSV or JSON")
    modified_at = datetime.fromisoformat(
        _text(entry, "modified_at").replace("Z", "+00:00")
    )
    nanoseconds = int(modified_at.timestamp() * 1_000_000_000)
    os.utime(path, ns=(nanoseconds, nanoseconds))


def _aggregate_oracle(fixture: CommerceFixture) -> AggregateOracle:
    with sqlite3.connect(fixture.database_path) as connection:
        row = connection.execute(
            _revenue_cte() + """
            SELECT COUNT(DISTINCT customer_id),
                   COALESCE(SUM(payment_cents - refunded_cents), 0)
            FROM paid_revenue
            """,
            (fixture.reporting_start, fixture.reporting_end_exclusive),
        ).fetchone()
    assert row is not None
    return AggregateOracle(customer_count=int(row[0]), net_revenue_cents=int(row[1]))


def _leading_region_oracle(fixture: CommerceFixture) -> RegionOracle:
    with sqlite3.connect(fixture.database_path) as connection:
        row = connection.execute(
            _revenue_cte() + """
            SELECT region_name, SUM(payment_cents - refunded_cents) AS net_cents
            FROM paid_revenue
            GROUP BY region_name
            ORDER BY net_cents DESC, region_name ASC
            LIMIT 1
            """,
            (fixture.reporting_start, fixture.reporting_end_exclusive),
        ).fetchone()
    if row is None:
        raise AssertionError("commerce fixture must produce a leading region")
    return RegionOracle(region_name=str(row[0]), net_revenue_cents=int(row[1]))


def _plan_breakdown_oracle(
    fixture: CommerceFixture,
) -> dict[str, AggregateOracle]:
    with sqlite3.connect(fixture.database_path) as connection:
        rows = connection.execute(
            _revenue_cte() + """
            SELECT plan, COUNT(DISTINCT customer_id),
                   SUM(payment_cents - refunded_cents)
            FROM paid_revenue
            GROUP BY plan
            ORDER BY plan
            """,
            (fixture.reporting_start, fixture.reporting_end_exclusive),
        ).fetchall()
    return {
        str(plan): AggregateOracle(int(customer_count), int(net_cents))
        for plan, customer_count, net_cents in rows
    }


def _enterprise_refund_oracle(fixture: CommerceFixture) -> tuple[int, int]:
    with sqlite3.connect(fixture.database_path) as connection:
        row = connection.execute(
            _revenue_cte() + """
            SELECT COUNT(DISTINCT customer_id), COALESCE(SUM(refunded_cents), 0)
            FROM paid_revenue
            WHERE plan = 'enterprise' AND refunded_cents > 0
            """,
            (fixture.reporting_start, fixture.reporting_end_exclusive),
        ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


def _revenue_cte() -> str:
    return """
        WITH refund_totals AS (
            SELECT payment_id, SUM(amount_cents) AS refunded_cents
            FROM refunds
            GROUP BY payment_id
        ),
        paid_revenue AS (
            SELECT c.id AS customer_id,
                   c.plan AS plan,
                   r.name AS region_name,
                   p.amount_cents AS payment_cents,
                   COALESCE(rt.refunded_cents, 0) AS refunded_cents
            FROM customers AS c
            JOIN regions AS r ON r.id = c.region_id
            JOIN orders AS o ON o.customer_id = c.id
            JOIN payments AS p ON p.order_id = o.id
            LEFT JOIN refund_totals AS rt ON rt.payment_id = p.id
            WHERE c.lifecycle_status = 'active'
              AND o.status = 'paid'
              AND p.status = 'succeeded'
              AND substr(o.ordered_at, 1, 10) >= ?
              AND substr(o.ordered_at, 1, 10) < ?
        )
    """


def _newest_customer_export(fixture: CommerceFixture) -> Path:
    manifest = load_manifest()
    entries = tuple(
        cast(Mapping[str, object], entry)
        for entry in _sequence(manifest, "files")
        if isinstance(entry, Mapping) and entry.get("kind") == "customer_export"
    )
    candidates = tuple(
        fixture.files_root / _text(entry, "filename") for entry in entries
    )
    if len(candidates) < 2:
        raise AssertionError("commerce fixture requires two customer exports")
    freshness = tuple((path.stat().st_mtime_ns, path) for path in candidates)
    greatest = max(modified_at for modified_at, _ in freshness)
    newest = tuple(path for modified_at, path in freshness if modified_at == greatest)
    if len(newest) != 1:
        raise AssertionError(
            "customer export freshness is ambiguous at the greatest modified time"
        )
    return newest[0]


def _comparison_oracle(
    fixture: CommerceFixture,
    newest: Path,
) -> tuple[tuple[Discrepancy, ...], str, int]:
    file_dataset, database_dataset = build_customer_comparison_datasets(
        fixture,
        newest,
    )
    result = compare_tabular_datasets(
        file_dataset,
        database_dataset,
        key_columns=("id",),
        compare_columns=_CUSTOMER_VALUE_COLUMNS,
        key_normalization=CROSS_SOURCE_KEY_NORMALIZATION,
    )
    policy = _mapping(result.payload, "comparison_policy")
    key_normalization = policy.get("key_normalization")
    schema_version = policy.get("schema_version")
    if (
        not isinstance(key_normalization, str)
        or key_normalization != CROSS_SOURCE_KEY_NORMALIZATION
    ):
        raise AssertionError("production comparison used an unexpected key policy")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != TABULAR_COMPARISON_POLICY_SCHEMA_VERSION
    ):
        raise AssertionError("production comparison policy schema is unexpected")
    return (
        normalize_comparison_discrepancies(
            result,
            file_evidence_id=file_dataset.evidence_id,
        ),
        key_normalization,
        schema_version,
    )


def build_customer_comparison_datasets(
    fixture: CommerceFixture,
    export: Path,
) -> tuple[TabularEvidenceDataset, TabularEvidenceDataset]:
    """Build native-typed, authoritative inputs for the production comparator."""

    if export.parent != fixture.files_root or not export.is_file():
        raise AssertionError("comparison export must be a current fixture file")
    with export.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _CUSTOMER_COMPARE_COLUMNS:
            raise AssertionError(
                "customer export columns do not match the oracle contract"
            )
        file_rows = tuple(dict(row) for row in reader)
    with sqlite3.connect(fixture.database_path) as connection:
        connection.row_factory = sqlite3.Row
        database_rows = tuple(
            {column: row[column] for column in _CUSTOMER_COMPARE_COLUMNS}
            for row in connection.execute(
                "SELECT id, name, plan, lifecycle_status, email FROM customers"
            )
        )

    return (
        TabularEvidenceDataset(
            operation_id="fixture-oracle-operation",
            evidence_id="fixture-customer-export-evidence",
            evidence_kind="data.file.read_result",
            source_id="fixture-local-files",
            source_revision=fixture.manifest_digest,
            resource_revisions=((export.name, _file_digest(export)),),
            columns=_CUSTOMER_COMPARE_COLUMNS,
            rows=file_rows,
            complete=True,
            truncation_reasons=(),
            row_limit=max(1, len(file_rows)),
            byte_limit=max(2, export.stat().st_size),
        ),
        TabularEvidenceDataset(
            operation_id="fixture-oracle-operation",
            evidence_id="fixture-customer-table-evidence",
            evidence_kind="data.sqlite.query_result",
            source_id="fixture-sqlite",
            source_revision=fixture.manifest_digest,
            resource_revisions=(
                ("main.customers", _file_digest(fixture.database_path)),
            ),
            columns=_CUSTOMER_COMPARE_COLUMNS,
            rows=database_rows,
            complete=True,
            truncation_reasons=(),
            row_limit=max(1, len(database_rows)),
            byte_limit=max(2, fixture.database_path.stat().st_size),
        ),
    )


def normalize_comparison_discrepancies(
    comparison: TabularComparisonResult | Mapping[str, object],
    *,
    file_evidence_id: str,
) -> tuple[Discrepancy, ...]:
    """Normalize production or persisted live output without weakening semantics."""

    payload: Mapping[str, object]
    if isinstance(comparison, TabularComparisonResult):
        payload = comparison.payload
        discrepancies: Sequence[Mapping[str, object]] = comparison.discrepancies
    else:
        payload = comparison
        raw_discrepancies = payload.get("discrepancy_sample")
        if not isinstance(raw_discrepancies, Sequence) or isinstance(
            raw_discrepancies,
            (str, bytes),
        ):
            raise AssertionError("persisted comparison lacks a discrepancy sample")
        if any(not isinstance(item, Mapping) for item in raw_discrepancies):
            raise AssertionError("persisted comparison discrepancies must be objects")
        discrepancies = cast(
            Sequence[Mapping[str, object]],
            raw_discrepancies,
        )

    policy = _mapping(payload, "comparison_policy")
    if policy.get("key_normalization") != CROSS_SOURCE_KEY_NORMALIZATION:
        raise AssertionError("live comparison did not use the declared key policy")
    if policy.get("schema_version") != TABULAR_COMPARISON_POLICY_SCHEMA_VERSION:
        raise AssertionError("live comparison policy schema is unexpected")
    left = _mapping(payload, "left")
    right = _mapping(payload, "right")
    left_evidence_id = left.get("evidence_id")
    right_evidence_id = right.get("evidence_id")
    if file_evidence_id == left_evidence_id:
        file_is_left = True
    elif file_evidence_id == right_evidence_id:
        file_is_left = False
    else:
        raise AssertionError("file evidence is not a comparison input")

    normalized = tuple(
        _normalize_discrepancy(item, file_is_left=file_is_left)
        for item in discrepancies
    )
    return tuple(sorted(normalized, key=_discrepancy_sort_key))


def _normalize_discrepancy(
    item: Mapping[str, object],
    *,
    file_is_left: bool,
) -> Discrepancy:
    kind = item.get("kind")
    if not isinstance(kind, str) or kind not in _PRODUCTION_DISCREPANCY_KINDS:
        raise AssertionError("comparison emitted an unsupported discrepancy kind")

    if kind == "invalid_key":
        source = _normalized_source(item, file_is_left=file_is_left)
        return Discrepancy(
            kind=kind,
            customer_id=None,
            source=source,
            missing_columns=_text_tuple(item.get("missing_columns")),
            null_columns=_text_tuple(item.get("null_columns")),
        )

    customer_id = _normalized_customer_id(item)
    if kind == "duplicate_key":
        row_indexes = item.get("row_indexes")
        if not isinstance(row_indexes, Sequence) or isinstance(
            row_indexes,
            (str, bytes),
        ):
            raise AssertionError("duplicate discrepancy lacks row indexes")
        return Discrepancy(
            kind=kind,
            customer_id=customer_id,
            source=_normalized_source(item, file_is_left=file_is_left),
            duplicate_count=len(row_indexes),
        )

    if kind in {"left_only", "right_only"}:
        normalized_kind = kind
        if not file_is_left:
            normalized_kind = "right_only" if kind == "left_only" else "left_only"
        return Discrepancy(kind=normalized_kind, customer_id=customer_id)

    column = item.get("column")
    if not isinstance(column, str) or not column:
        raise AssertionError("value discrepancy lacks a compared column")
    file_side = "left" if file_is_left else "right"
    database_side = "right" if file_is_left else "left"
    file_present = _boolean(item, f"{file_side}_present")
    database_present = _boolean(item, f"{database_side}_present")
    return Discrepancy(
        kind=kind,
        customer_id=customer_id,
        column=column,
        file_value=item.get(f"{file_side}_value") if file_present else None,
        database_value=(
            item.get(f"{database_side}_value") if database_present else None
        ),
        file_present=file_present,
        database_present=database_present,
        file_type=_required_item_text(item, f"{file_side}_type"),
        database_type=_required_item_text(item, f"{database_side}_type"),
    )


def _normalized_source(
    item: Mapping[str, object],
    *,
    file_is_left: bool,
) -> str:
    side = item.get("side")
    if side not in {"left", "right"}:
        raise AssertionError("side-specific discrepancy lacks its source side")
    is_file = (side == "left") == file_is_left
    return "file" if is_file else "database"


def _normalized_customer_id(item: Mapping[str, object]) -> str:
    key = _mapping(item, "normalized_key")
    value = key.get("id")
    if value is None:
        raise AssertionError("customer discrepancy lacks a normalized id")
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _discrepancy_sort_key(item: Discrepancy) -> tuple[str, str, str, str]:
    return (
        item.customer_id or "",
        item.kind,
        item.column or "",
        repr(item),
    )


def _boolean(value: Mapping[str, object], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise AssertionError(f"comparison discrepancy {key} must be boolean")
    return item


def _required_item_text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise AssertionError(f"comparison discrepancy {key} must be text")
    return item


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AssertionError("comparison discrepancy columns must be a sequence")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise AssertionError("comparison discrepancy columns must be strings")
    return cast(tuple[str, ...], result)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"fixture manifest {key} must be an object")
    return cast(Mapping[str, object], item)


def _sequence(value: Mapping[str, object], key: str) -> Sequence[object]:
    item = value.get(key)
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
        raise ValueError(f"fixture manifest {key} must be an array")
    return cast(Sequence[object], item)


def _text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"fixture manifest {key} must be non-empty text")
    return item


def _safe_filename(value: str) -> str:
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("fixture filename must be one relative leaf")
    return value


__all__ = [
    "AggregateOracle",
    "CommerceFixture",
    "CommerceOracles",
    "CROSS_SOURCE_KEY_NORMALIZATION",
    "Discrepancy",
    "EXPECTED_FIXTURE_VERSION",
    "EXPECTED_MANIFEST_SHA256",
    "MANIFEST_PATH",
    "RegionOracle",
    "build_commerce_fixture",
    "build_customer_comparison_datasets",
    "canonical_manifest_bytes",
    "compute_oracles",
    "load_manifest",
    "manifest_digest",
    "money_text",
    "normalize_comparison_discrepancies",
]
