"""`daita demo`: deterministic sample data and an isolated demo agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from daita import Agent, sample_data
from daita.sample_data import (
    SAMPLE_DATABASE_FILENAME,
    SAMPLE_TABLES,
    build_sample_database,
    sample_tables,
)
from tests.support.cli import invoke
from tests.support.workspace import workspace_for

# Changing the generator changes this digest on purpose: review the new data,
# then update it. A difference on one platform only means the data is not
# deterministic there.
SAMPLE_DIGEST = "0d506486a5d1f6b4f24f3f9614db94e451b91870d79a9dd76ef96708faca2caa"


def _digest(tables: object) -> str:
    encoded = json.dumps(tables, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rows(path: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            ).fetchall()
            for table in SAMPLE_TABLES
        }


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("sample")
    return build_sample_database(directory / SAMPLE_DATABASE_FILENAME)


def _scalar(path: Path, query: str) -> object:
    with sqlite3.connect(path) as connection:
        return connection.execute(query).fetchone()[0]


@pytest.mark.unit
def test_sample_rows_are_pinned_across_runs_and_platforms() -> None:
    assert _digest(sample_tables()) == SAMPLE_DIGEST
    assert _digest(sample_tables()) == SAMPLE_DIGEST


@pytest.mark.unit
def test_built_database_holds_exactly_the_generated_rows(
    database: Path, tmp_path: Path
) -> None:
    again = build_sample_database(tmp_path / SAMPLE_DATABASE_FILENAME)

    expected = {
        table: [tuple(row) for row in rows] for table, rows in sample_tables().items()
    }
    assert _rows(database) == expected
    assert _rows(again) == expected
    assert database.stat().st_mode & 0o777 == 0o600
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        SAMPLE_DATABASE_FILENAME
    ]


@pytest.mark.unit
def test_sample_database_is_a_small_business_with_real_foreign_keys(
    database: Path,
) -> None:
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        references = {
            (table, row[2])
            for table in SAMPLE_TABLES
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        }
        span = connection.execute(
            "SELECT min(ordered_at), max(ordered_at) FROM orders"
        ).fetchone()
    assert references == {
        ("accounts", "customers"),
        ("orders", "accounts"),
        ("order_items", "orders"),
        ("order_items", "products"),
        ("invoices", "orders"),
        ("payments", "invoices"),
        ("support_tickets", "customers"),
    }
    assert span[0] < "2023-04-01" and span[1] > "2025-12-01"
    assert database.stat().st_size < 2 * 1024 * 1024


@pytest.mark.unit
def test_planted_regional_revenue_dip_shows_in_a_quarterly_profile(
    database: Path,
) -> None:
    def quarter(region: str, first: str, last: str) -> float:
        value = _scalar(
            database,
            "SELECT sum(o.total) FROM orders o "
            "JOIN accounts a ON a.id = o.account_id "
            "JOIN customers c ON c.id = a.customer_id "
            f"WHERE o.status = 'completed' AND c.region = '{region}' "
            f"AND o.ordered_at BETWEEN '{first}' AND '{last}'",
        )
        assert isinstance(value, float)
        return value

    europe_q2 = quarter("Europe", "2025-04-01", "2025-06-30")
    europe_q3 = quarter("Europe", "2025-07-01", "2025-09-30")
    assert europe_q3 < 0.6 * europe_q2
    for region in ("North America", "Asia Pacific", "Latin America"):
        assert quarter(region, "2025-07-01", "2025-09-30") > 0.7 * quarter(
            region, "2025-04-01", "2025-06-30"
        )


@pytest.mark.unit
def test_planted_duplicates_late_payers_and_stale_accounts_are_discoverable(
    database: Path,
) -> None:
    duplicated_domains = _scalar(
        database,
        "SELECT count(*) FROM (SELECT substr(email, instr(email, '@')) AS domain "
        "FROM customers GROUP BY domain HAVING count(*) > 1)",
    )
    late_payers = _scalar(
        database,
        "SELECT count(*) FROM (SELECT a.customer_id FROM payments p "
        "JOIN invoices i ON i.id = p.invoice_id "
        "JOIN orders o ON o.id = i.order_id "
        "JOIN accounts a ON a.id = o.account_id "
        "WHERE p.method != 'refund' GROUP BY a.customer_id "
        "HAVING avg(julianday(p.paid_at) - julianday(i.due_at)) > 30)",
    )
    stale_active_accounts = _scalar(
        database,
        "SELECT count(*) FROM accounts a WHERE a.status = 'active' "
        "AND NOT EXISTS (SELECT 1 FROM orders o WHERE o.account_id = a.id "
        "AND o.status = 'completed' AND o.ordered_at > date("
        "(SELECT value FROM dataset_info WHERE key = 'as_of'), '-90 days'))",
    )
    assert duplicated_domains == 6
    assert isinstance(late_payers, int) and late_payers >= 5
    assert isinstance(stale_active_accounts, int) and stale_active_accounts >= 10


@pytest.mark.unit
def test_suggested_questions_cover_the_four_first_session_moves() -> None:
    assert len(sample_data.SUGGESTED_QUESTIONS) == 4
    assert sample_data.LEARN_EXAMPLE.startswith("/learn ")


def _demo(root: Path, *argv: str) -> tuple[int, str, str]:
    work = root.parent / "work"
    work.mkdir(exist_ok=True)
    return invoke(
        ["--no-host", "--root", str(root), "--workspace", str(work), "demo", *argv]
    )


async def _sources(root: Path, name: str) -> list[tuple[str, str]]:
    agent = await Agent.open(name, root=root, workspace=workspace_for(root))
    try:
        return [
            (item.display_name, str(item.configuration.get("path")))
            for item in await agent.list_sources()
            if item.active
        ]
    finally:
        await agent.close()


@pytest.mark.integration
def test_demo_creates_an_isolated_agent_over_the_sample_data(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "state"
    other = root.parent / "drop"
    other.mkdir()
    (other / "notes.csv").write_text("a\n1\n", encoding="utf-8")
    base = ["--no-host", "--root", str(root)]
    assert invoke([*base, "create", "atlas"])[0] == 0
    assert invoke([*base, "attach", "atlas", "folder", str(other)])[0] == 0
    before = asyncio.run(_sources(root, "atlas"))

    code, stdout, stderr = _demo(root, "--json")

    assert (code, stderr) == (0, "")
    result = json.loads(stdout)
    database = root / "demo" / SAMPLE_DATABASE_FILENAME
    assert result["agent"] == "demo"
    assert result["database"] == str(database)
    assert result["source"] == "sample_business"
    assert result["model_configured"] is False
    assert result["open_command"] == f"daita --root {root} chat demo"
    assert len(result["questions"]) == 4
    assert _digest(_rows(database)) == SAMPLE_DIGEST
    assert asyncio.run(_sources(root, "demo")) == [("sample_business", str(database))]
    assert asyncio.run(_sources(root, "atlas")) == before
    assert asyncio.run(Agent.list(root=root)) == ("atlas", "demo")


@pytest.mark.integration
def test_demo_is_reused_on_a_second_run_and_reset_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "state"
    first = json.loads(_demo(root, "--json")[1])

    second = json.loads(_demo(root, "--json")[1])
    assert second["agent_id"] == first["agent_id"]

    database = Path(first["database"])
    rows = _rows(database)
    resets = []
    for _ in range(2):
        code, stdout, stderr = _demo(root, "--reset", "--json")
        assert (code, stderr) == (0, "")
        resets.append(json.loads(stdout))
    assert len({first["agent_id"], *(item["agent_id"] for item in resets)}) == 3
    assert _rows(database) == rows
    assert asyncio.run(_sources(root, "demo")) == [("sample_business", str(database))]
    assert asyncio.run(Agent.list(root=root)) == ("demo",)
    assert sorted(item.name for item in database.parent.iterdir()) == [
        "agent.json",
        SAMPLE_DATABASE_FILENAME,
    ]


@pytest.mark.integration
def test_demo_never_takes_over_an_agent_it_did_not_create(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "state"
    assert invoke(["--no-host", "--root", str(root), "create", "demo"])[0] == 0

    for argv in ((), ("--reset",)):
        code, stdout, stderr = _demo(root, *argv)
        assert (code, stdout) == (1, "")
        assert "was not created by `daita demo`" in json.loads(stderr)["error"]
        assert asyncio.run(_sources(root, "demo")) == []
    assert not (root / "demo" / SAMPLE_DATABASE_FILENAME).exists()


@pytest.mark.integration
def test_demo_prints_a_guide_with_the_questions_and_model_setup(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "state"

    code, stdout, stderr = _demo(root)

    assert (code, stderr) == (0, "")
    assert f"Open it:  daita --root {root} chat demo" in stdout
    assert "/model" in stdout
    for number, question in enumerate(sample_data.SUGGESTED_QUESTIONS, start=1):
        assert f"  {number}. {question}" in stdout
    assert sample_data.LEARN_EXAMPLE in stdout
    assert "daita demo --reset" in stdout
