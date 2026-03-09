"""
Live SQL focus pushdown integration tests.

Each test class covers one plugin (PostgreSQL, MySQL, Snowflake) and verifies:
  1. SQL pushdown — the focus DSL is compiled into SQL clauses sent to the DB.
  2. Execution correctness — the compiled SQL is run against a real in-memory
     SQLite database seeded with the full dataset. The mock does not return
     hardcoded data; it executes the pushed-down SQL for real.
  3. LLM correctness — the agent produces an accurate answer from the data
     the DB actually returned, confirming pushdown doesn't degrade quality.

No real database connection is required. Each plugin's query() method is
replaced with one that executes received SQL in SQLite (after normalising
placeholder styles), giving us genuine SQL semantics in tests.

Run with:
    OPENAI_API_KEY=sk-... pytest tests/unit/test_focus_sql_live.py -v -s -m requires_llm
"""
import os
import re
import sqlite3
import json
import pytest

from daita.core.tools import AgentTool
from daita.agents.agent import Agent


# ── Full datasets (wide, realistic) ───────────────────────────────────────────

ORDERS = [
    {"order_id": "ORD-001", "customer_id": "CUST-042", "customer_name": "Lena Fischer",
     "email": "lena@example.com", "product": "Pro Plan", "category": "subscription",
     "amount": 299.00, "status": "completed", "region": "EU",
     "created_at": "2024-03-01T09:12:00Z", "payment_method": "credit_card",
     "refunded": 0, "notes": ""},
    {"order_id": "ORD-002", "customer_id": "CUST-007", "customer_name": "James Park",
     "email": "james@example.com", "product": "Starter Plan", "category": "subscription",
     "amount": 49.00, "status": "completed", "region": "US",
     "created_at": "2024-03-02T14:05:00Z", "payment_method": "paypal",
     "refunded": 0, "notes": ""},
    {"order_id": "ORD-003", "customer_id": "CUST-099", "customer_name": "Sara Nkosi",
     "email": "sara@example.com", "product": "Enterprise", "category": "subscription",
     "amount": 999.00, "status": "refunded", "region": "EU",
     "created_at": "2024-03-03T11:00:00Z", "payment_method": "credit_card",
     "refunded": 1, "notes": "Customer requested refund"},
    {"order_id": "ORD-004", "customer_id": "CUST-011", "customer_name": "Hiroshi Tanaka",
     "email": "hiroshi@example.com", "product": "Pro Plan", "category": "subscription",
     "amount": 299.00, "status": "completed", "region": "APAC",
     "created_at": "2024-03-04T02:45:00Z", "payment_method": "credit_card",
     "refunded": 0, "notes": ""},
    {"order_id": "ORD-005", "customer_id": "CUST-055", "customer_name": "Maria Garcia",
     "email": "maria@example.com", "product": "Starter Plan", "category": "subscription",
     "amount": 49.00, "status": "pending", "region": "US",
     "created_at": "2024-03-05T17:22:00Z", "payment_method": "bank_transfer",
     "refunded": 0, "notes": "Awaiting payment confirmation"},
]

SESSIONS = [
    {"session_id": "S001", "user_id": "u100", "page": "pricing",  "duration_s": 145,
     "converted": 1, "browser": "Chrome", "device": "desktop", "referrer": "google",
     "country": "US", "bounce": 0, "scroll_depth_pct": 82, "ab_variant": "B"},
    {"session_id": "S002", "user_id": "u101", "page": "pricing",  "duration_s": 12,
     "converted": 0, "browser": "Safari", "device": "mobile",  "referrer": "twitter",
     "country": "UK", "bounce": 1, "scroll_depth_pct": 15, "ab_variant": "A"},
    {"session_id": "S003", "user_id": "u102", "page": "features", "duration_s": 90,
     "converted": 0, "browser": "Firefox", "device": "desktop", "referrer": "direct",
     "country": "DE", "bounce": 0, "scroll_depth_pct": 55, "ab_variant": "A"},
    {"session_id": "S004", "user_id": "u103", "page": "pricing",  "duration_s": 210,
     "converted": 1, "browser": "Chrome", "device": "desktop", "referrer": "blog",
     "country": "CA", "bounce": 0, "scroll_depth_pct": 100, "ab_variant": "B"},
    {"session_id": "S005", "user_id": "u104", "page": "pricing",  "duration_s": 8,
     "converted": 0, "browser": "Edge",   "device": "mobile",  "referrer": "email",
     "country": "US", "bounce": 1, "scroll_depth_pct": 5, "ab_variant": "A"},
    {"session_id": "S006", "user_id": "u105", "page": "pricing",  "duration_s": 180,
     "converted": 1, "browser": "Chrome", "device": "tablet",  "referrer": "google",
     "country": "AU", "bounce": 0, "scroll_depth_pct": 91, "ab_variant": "B"},
]

SALES = [
    {"sale_id": "SL001", "rep_id": "R10", "rep_name": "Alice Wong",
     "product_line": "Cloud",    "region": "EMEA", "amount": 12000.0,
     "quarter": "Q1", "deal_size": "mid",   "industry": "fintech",      "discount_pct": 5},
    {"sale_id": "SL002", "rep_id": "R11", "rep_name": "Carlos Ruiz",
     "product_line": "On-Prem",  "region": "AMER", "amount": 45000.0,
     "quarter": "Q1", "deal_size": "large", "industry": "healthcare",   "discount_pct": 10},
    {"sale_id": "SL003", "rep_id": "R12", "rep_name": "Priya Shah",
     "product_line": "Cloud",    "region": "EMEA", "amount": 8500.0,
     "quarter": "Q1", "deal_size": "small", "industry": "retail",       "discount_pct": 0},
    {"sale_id": "SL004", "rep_id": "R13", "rep_name": "Tom Braun",
     "product_line": "Services", "region": "EMEA", "amount": 22000.0,
     "quarter": "Q1", "deal_size": "mid",   "industry": "manufacturing","discount_pct": 8},
    {"sale_id": "SL005", "rep_id": "R14", "rep_name": "Yuki Ito",
     "product_line": "Cloud",    "region": "APAC", "amount": 31000.0,
     "quarter": "Q1", "deal_size": "large", "industry": "tech",         "discount_pct": 12},
    {"sale_id": "SL006", "rep_id": "R15", "rep_name": "Sophie Martin",
     "product_line": "On-Prem",  "region": "EMEA", "amount": 19000.0,
     "quarter": "Q1", "deal_size": "mid",   "industry": "fintech",      "discount_pct": 5},
    {"sale_id": "SL007", "rep_id": "R16", "rep_name": "David Kim",
     "product_line": "Services", "region": "AMER", "amount": 9000.0,
     "quarter": "Q1", "deal_size": "small", "industry": "retail",       "discount_pct": 0},
]


# ── SQLite execution engine ────────────────────────────────────────────────────

def _build_sqlite_db(data: list, table: str) -> sqlite3.Connection:
    """
    Create an in-memory SQLite database with *data* loaded into *table*.
    Types are inferred from the first non-null value in each column.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    if not data:
        return conn

    cols = list(data[0].keys())

    def _infer(col):
        for row in data:
            v = row.get(col)
            if v is None:
                continue
            if isinstance(v, bool):  return "INTEGER"
            if isinstance(v, int):   return "INTEGER"
            if isinstance(v, float): return "REAL"
            return "TEXT"
        return "TEXT"

    col_defs = ", ".join(f'"{c}" {_infer(c)}' for c in cols)
    conn.execute(f'CREATE TABLE "{table}" ({col_defs})')

    for row in data:
        vals = [
            (1 if row[c] is True else 0 if row[c] is False else row[c])
            for c in cols
        ]
        conn.execute(
            f'INSERT INTO "{table}" VALUES ({",".join("?" * len(cols))})', vals
        )

    conn.commit()
    return conn


def _make_sqlite_query_fn(data: list, table: str):
    """
    Return an async query() compatible with BaseDatabasePlugin that executes
    the received SQL in a real SQLite in-memory database.

    Placeholder styles are normalised before execution:
      $1, $2 … (PostgreSQL)  →  ?
      %s      (MySQL/Snowflake) →  ?
    """
    conn = _build_sqlite_db(data, table)

    async def _query(sql: str, params=None):
        sqlite_sql = re.sub(r'\$\d+|%s', '?', sql)
        cursor = conn.execute(sqlite_sql, params or [])
        return [dict(row) for row in cursor.fetchall()]

    return _query


# ── Mock plugin factories ──────────────────────────────────────────────────────

def _make_pg_plugin(data: list, table: str):
    from daita.plugins.postgresql import PostgreSQLPlugin

    class _MockPG(PostgreSQLPlugin):
        def __init__(self):
            self._pool = None
            self._connection = None
            self._client = None
            self._db = None
            self.config = {}
            self.timeout = 30
            self.max_retries = 3
            self.connection_string = "postgresql://mock"
            self.pool_config = {}
            self.captured_sql: list[str] = []

        async def connect(self): pass
        async def disconnect(self): pass

        async def query(self, sql, params=None):
            self.captured_sql.append(sql)
            return await _make_sqlite_query_fn(data, table)(sql, params)

    return _MockPG()


def _make_mysql_plugin(data: list, table: str):
    from daita.plugins.mysql import MySQLPlugin

    class _MockMySQL(MySQLPlugin):
        def __init__(self):
            self._pool = None
            self._connection = None
            self._client = None
            self._db = None
            self.config = {}
            self.timeout = 30
            self.max_retries = 3
            self.host = "mock"
            self.port = 3306
            self.user = "mock"
            self.password = ""
            self.db = "mock"
            self.connection_string = "mysql://mock"
            self.pool_config = {}
            self.captured_sql: list[str] = []

        async def connect(self): pass
        async def disconnect(self): pass

        async def query(self, sql, params=None):
            self.captured_sql.append(sql)
            return await _make_sqlite_query_fn(data, table)(sql, params)

    return _MockMySQL()


def _make_sf_plugin(data: list, table: str):
    from daita.plugins.snowflake import SnowflakePlugin

    class _MockSnowflake(SnowflakePlugin):
        def __init__(self):
            self._pool = None
            self._connection = None
            self._client = None
            self._db = None
            self.config = {}
            self.timeout = 30
            self.max_retries = 3
            self.captured_sql: list[str] = []

        async def connect(self): pass
        async def disconnect(self): pass

        async def query(self, sql, params=None):
            self.captured_sql.append(sql)
            return await _make_sqlite_query_fn(data, table)(sql, params)

    return _MockSnowflake()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _token_estimate(data) -> int:
    return len(json.dumps(data, default=str)) // 4


def _make_agent(tool: AgentTool) -> Agent:
    return Agent(
        name="SQLFocusTestAgent",
        llm_provider="openai",
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        tools=[tool],
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.requires_llm
class TestPostgreSQLFocusPushdown:
    """
    Scenario: SaaS orders table with billing addresses, emails, and timestamps.

    Focus pushes WHERE + SELECT into PostgreSQL SQL. The mock executes the
    received SQL in SQLite — so the result is derived from real execution,
    not a hardcoded fixture.

    Ground truth (computed from ORDERS):
      Completed orders: ORD-001 Pro $299, ORD-002 Starter $49, ORD-004 Pro $299
      Highest-revenue completed product: Pro Plan at $598
    """

    async def test_sql_pushdown_and_execution(self):
        """
        Verify pushdown structure AND that executing the compiled SQL against
        real data returns only completed orders with the projected columns.
        """
        plugin = _make_pg_plugin(ORDERS, "orders")
        focus = "status == 'completed' | SELECT order_id, product, amount"

        result = await plugin._tool_query({
            "sql": "SELECT * FROM orders",
            "focus": focus,
        })

        pushed_sql = plugin.captured_sql[0]
        rows = result["rows"]

        print(f"\n[PG pushdown] SQL → {pushed_sql}")
        print(f"[PG result]   {rows}")

        # SQL structure
        assert "WHERE" in pushed_sql
        assert '"order_id"' in pushed_sql
        assert '"product"' in pushed_sql
        assert '"amount"' in pushed_sql

        # Real execution result — only completed orders, only 3 projected columns
        assert len(rows) == 3, f"Expected 3 completed orders, got {len(rows)}"
        assert all(r["status"] != "refunded" for r in rows if "status" in r)
        assert all(set(r.keys()) == {"order_id", "product", "amount"} for r in rows), \
            f"Unexpected columns in result: {[set(r.keys()) for r in rows]}"

        # Token reduction
        raw_tok = _token_estimate(ORDERS)
        focused_tok = _token_estimate(rows)
        print(f"[PG tokens]   raw≈{raw_tok}  focused≈{focused_tok}  (-{round((1-focused_tok/raw_tok)*100,1)}%)")

    async def test_llm_answers_correctly(self):
        """
        Full pipeline: mock PG executes pushed-down SQL in SQLite → LLM
        receives only the 3 relevant columns → must identify Pro Plan at $598.
        """
        plugin = _make_pg_plugin(ORDERS, "orders")

        tool = AgentTool(
            name="postgres_query",
            description="Run a SELECT on the orders table. Use focus to filter/project at DB level.",
            parameters={
                "type": "object",
                "properties": {
                    "sql":   {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["sql"],
            },
            handler=plugin._tool_query,
            source="plugin",
        )

        result = await _make_agent(tool).run_detailed(
            "Use postgres_query with SQL \"SELECT * FROM orders\" "
            "and focus \"status == 'completed' | SELECT order_id, product, amount\" "
            "to find which product generated the most revenue from completed orders. "
            "State the product name and exact total."
        )

        answer = result["result"]
        print(f"\n[PG LLM] tokens={result.get('tokens',{}).get('total_tokens')}  answer={answer[:200]}")

        assert "Pro Plan" in answer, f"Expected 'Pro Plan': {answer}"
        assert "598" in answer or "598.00" in answer, f"Expected $598: {answer}"


@pytest.mark.requires_llm
class TestMySQLFocusPushdown:
    """
    Scenario: Web analytics sessions table with browser, device, scroll depth etc.

    Focus pushes a compound filter (page + duration) and SELECT projection into
    MySQL SQL. The mock runs the compiled SQL in SQLite with the full dataset.

    Ground truth (computed from SESSIONS):
      Pricing page, duration > 30s: S001(converted=1), S004(converted=1), S006(converted=1)
      All 3 converted → conversion rate = 100% of that segment, count = 3
    """

    async def test_sql_pushdown_and_execution(self):
        """
        Verify MySQL backtick quoting and compound filter pushdown. Confirm
        SQLite execution returns the right sessions — not a hardcoded list.
        """
        plugin = _make_mysql_plugin(SESSIONS, "sessions")
        focus = "page == 'pricing' and duration_s > 30 | SELECT session_id, user_id, converted, duration_s"

        result = await plugin._tool_query({
            "sql": "SELECT * FROM sessions",
            "focus": focus,
        })

        pushed_sql = plugin.captured_sql[0]
        rows = result["rows"]

        print(f"\n[MySQL pushdown] SQL → {pushed_sql}")
        print(f"[MySQL result]   {rows}")

        # SQL structure: MySQL uses backtick quoting
        assert "WHERE" in pushed_sql
        assert "`session_id`" in pushed_sql
        assert "`converted`" in pushed_sql
        assert "AND" in pushed_sql, "Compound filter should use AND"

        # Real execution: only pricing-page sessions with duration > 30s
        assert len(rows) == 3, f"Expected 3 sessions (S001, S004, S006), got {len(rows)}: {rows}"
        session_ids = {r["session_id"] for r in rows}
        assert session_ids == {"S001", "S004", "S006"}, f"Wrong sessions: {session_ids}"

        # All returned sessions should have converted=1 (ground truth from data)
        assert all(r["converted"] == 1 for r in rows), \
            f"All pricing+duration>30 sessions should have converted: {rows}"

    async def test_llm_answers_correctly(self):
        """
        Full pipeline: mock MySQL executes pushed-down SQL in SQLite → LLM
        counts 3 converting sessions correctly.
        """
        plugin = _make_mysql_plugin(SESSIONS, "sessions")

        tool = AgentTool(
            name="mysql_query",
            description="Query the sessions table. Use focus to filter/project at DB level.",
            parameters={
                "type": "object",
                "properties": {
                    "sql":   {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["sql"],
            },
            handler=plugin._tool_query,
            source="plugin",
        )

        result = await _make_agent(tool).run_detailed(
            "Use mysql_query with SQL \"SELECT * FROM sessions\" "
            "and focus \"page == 'pricing' and duration_s > 30 | SELECT session_id, user_id, converted, duration_s\" "
            "to find how many users who spent more than 30 seconds on the pricing page converted. "
            "Give the exact count."
        )

        answer = result["result"]
        print(f"\n[MySQL LLM] tokens={result.get('tokens',{}).get('total_tokens')}  answer={answer[:200]}")

        assert "3" in answer, f"Expected count of 3 in answer: {answer}"


@pytest.mark.requires_llm
class TestSnowflakeFocusPushdown:
    """
    Scenario: Snowflake sales data warehouse with rep names, industries, discounts.

    Focus pushes a region filter + GROUP BY aggregation into Snowflake SQL so
    the DB does the rollup. The mock executes compiled SQL in SQLite.

    Ground truth (computed from SALES, EMEA rows only):
      Cloud    → SL001 $12,000 + SL003 $8,500  = $20,500
      Services → SL004 $22,000
      On-Prem  → SL006 $19,000
      Top: Services at $22,000
    """

    async def test_sql_pushdown_and_execution(self):
        """
        Verify GROUP BY + SUM aggregate are pushed into Snowflake SQL and that
        SQLite execution of that SQL produces the correct aggregated totals.
        """
        plugin = _make_sf_plugin(SALES, "sales")
        focus = "region == 'EMEA' | GROUP BY product_line | SELECT product_line, SUM(amount) AS total_sales | ORDER BY total_sales DESC"

        result = await plugin._tool_query({
            "sql": "SELECT * FROM sales",
            "focus": focus,
        })

        pushed_sql = plugin.captured_sql[0]
        rows = result["rows"]

        print(f"\n[SF pushdown] SQL → {pushed_sql}")
        print(f"[SF result]   {rows}")

        # SQL structure
        assert "GROUP BY" in pushed_sql
        assert "SUM(" in pushed_sql
        assert '"total_sales"' in pushed_sql
        assert "WHERE" in pushed_sql
        assert "ORDER BY" in pushed_sql

        # Real execution: 3 EMEA product lines in descending revenue order
        assert len(rows) == 3, f"Expected 3 EMEA product lines, got {len(rows)}: {rows}"

        totals = {r["product_line"]: r["total_sales"] for r in rows}
        assert abs(totals["Cloud"]    - 20500.0) < 0.01, f"Cloud total wrong: {totals}"
        assert abs(totals["Services"] - 22000.0) < 0.01, f"Services total wrong: {totals}"
        assert abs(totals["On-Prem"]  - 19000.0) < 0.01, f"On-Prem total wrong: {totals}"

        # Ordered DESC by total_sales → Services first
        assert rows[0]["product_line"] == "Services", \
            f"Expected Services first (highest), got: {rows[0]}"

        raw_tok = _token_estimate(SALES)
        focused_tok = _token_estimate(rows)
        print(f"[SF tokens]   raw≈{raw_tok}  focused≈{focused_tok}  (-{round((1-focused_tok/raw_tok)*100,1)}%)")

    async def test_llm_answers_correctly(self):
        """
        Full pipeline: mock Snowflake executes GROUP BY SQL in SQLite → LLM
        identifies Services as top EMEA product line at $22,000.
        """
        plugin = _make_sf_plugin(SALES, "sales")

        tool = AgentTool(
            name="snowflake_query",
            description="Query the Snowflake sales warehouse. Use focus to push clauses into SQL.",
            parameters={
                "type": "object",
                "properties": {
                    "sql":   {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["sql"],
            },
            handler=plugin._tool_query,
            source="plugin",
        )

        result = await _make_agent(tool).run_detailed(
            "Use snowflake_query with SQL \"SELECT * FROM sales\" "
            "and focus \"region == 'EMEA' | GROUP BY product_line | SELECT product_line, SUM(amount) AS total_sales | ORDER BY total_sales DESC\" "
            "to find the top-revenue product line in EMEA. State the product line and exact total."
        )

        answer = result["result"]
        print(f"\n[SF LLM] tokens={result.get('tokens',{}).get('total_tokens')}  answer={answer[:200]}")

        assert "Services" in answer, f"Expected 'Services' as top EMEA line: {answer}"
        assert "22000" in answer or "22,000" in answer, f"Expected $22,000: {answer}"
