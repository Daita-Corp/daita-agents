"""Build the built-in sample database: a small, deterministic B2B business.

``daita demo`` attaches it so a first run needs no database of one's own. The
data comes from a fixed seed and a fixed calendar, so every run on every
platform produces the same rows, and generation needs only the standard library.

The business sells software and services to companies in four regions. A few
things are planted on purpose so a first session has something to find:

- European revenue drops sharply in the third quarter of 2025;
- a handful of customers were entered twice under slightly different names;
- a small group of customers pays its invoices well past the due date;
- some accounts still marked ``active`` have stopped ordering, so "active
  customer" is a business term worth teaching rather than a column to trust.
"""

from __future__ import annotations

import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SAMPLE_DATABASE_FILENAME = "sample_business.sqlite"
SAMPLE_AS_OF = date(2025, 12, 31)
_HISTORY_START = date(2023, 1, 1)
_SEED = 20_251_231

SUGGESTED_QUESTIONS = (
    "How do the tables in sample_business relate? Walk me from a customer to "
    "their payments.",
    "Profile the customers, orders and invoices tables. Does anything look off?",
    "Which region's revenue changed most between Q2 and Q3 2025, and which "
    "products drove the change?",
    "Every Monday, list invoices more than 30 days past due, grouped by customer.",
)
LEARN_EXAMPLE = (
    "/learn An active customer has an account with status 'active' and a "
    "completed order in the 90 days before the data's as_of date in dataset_info."
)

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE dataset_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    region TEXT NOT NULL,
    industry TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE accounts (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    plan TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'churned')),
    opened_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    ordered_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'refunded')),
    total REAL NOT NULL
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE invoices (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL UNIQUE REFERENCES orders(id),
    issued_at TEXT NOT NULL,
    due_at TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'open', 'overdue', 'refunded'))
);
CREATE TABLE payments (
    id INTEGER PRIMARY KEY,
    invoice_id INTEGER NOT NULL REFERENCES invoices(id),
    paid_at TEXT NOT NULL,
    amount REAL NOT NULL,
    method TEXT NOT NULL
);
CREATE TABLE support_tickets (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    priority TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed'))
);
"""

# Insertion order; every table's foreign keys point at an earlier table.
SAMPLE_TABLES = (
    "dataset_info",
    "customers",
    "accounts",
    "products",
    "orders",
    "order_items",
    "invoices",
    "payments",
    "support_tickets",
)

_REGIONS = ("North America", "Europe", "Asia Pacific", "Latin America")
_REGION_WEIGHTS = (35, 30, 20, 15)
_DIP_REGION = "Europe"
_DIP_MONTHS = ((2025, 7), (2025, 8), (2025, 9))
_DIP_SKIP_PROBABILITY = 0.65
_INDUSTRIES = (
    "Retail",
    "Healthcare",
    "Manufacturing",
    "Logistics",
    "Financial services",
    "Education",
    "Media",
    "Hospitality",
)
_NAME_STEMS = (
    "Northwind",
    "Bluepeak",
    "Cedarline",
    "Harborview",
    "Ironleaf",
    "Brightwater",
    "Stonebridge",
    "Silverpine",
    "Redfern",
    "Oakhurst",
    "Lumen",
    "Granite",
    "Meridian",
    "Kestrel",
    "Westbrook",
    "Summit",
    "Crescent",
    "Fairfield",
    "Ashgrove",
    "Pioneer",
    "Riverstone",
    "Evergreen",
    "Highland",
    "Copperfield",
    "Larkspur",
    "Tidewater",
    "Foxglove",
    "Marlowe",
    "Quarry",
    "Juniper",
)
_NAME_KINDS = (
    "Analytics",
    "Logistics",
    "Foods",
    "Health",
    "Systems",
    "Retail",
    "Labs",
    "Partners",
    "Freight",
    "Media",
)
_NAME_SUFFIXES = ("Inc", "Ltd", "LLC", "GmbH", "Co")
_CUSTOMER_COUNT = 180
_DUPLICATE_COUNT = 6
_LATE_PAYER_COUNT = 9
_PLANS = ("Starter", "Growth", "Enterprise")
_PLAN_WEIGHTS = (50, 35, 15)
# Monthly order count weights for 0, 1, 2 and 3 orders, by plan.
_ORDER_WEIGHTS = {
    "Starter": (55, 35, 9, 1),
    "Growth": (35, 40, 20, 5),
    "Enterprise": (15, 40, 30, 15),
}
_MAX_QUANTITY = {"Starter": 5, "Growth": 12, "Enterprise": 30}
_PRODUCTS = (
    ("Core platform seat", "Subscriptions", 4_900),
    ("Analytics add-on", "Subscriptions", 2_900),
    ("Data connector", "Subscriptions", 1_900),
    ("Premium support", "Subscriptions", 9_900),
    ("Audit log retention", "Subscriptions", 1_500),
    ("Onboarding workshop", "Services", 120_000),
    ("Custom integration", "Services", 350_000),
    ("Training session", "Services", 45_000),
    ("Health check", "Services", 80_000),
    ("Report pack", "Content", 7_500),
    ("Industry benchmark", "Content", 19_900),
    ("Template library", "Content", 3_900),
    ("Edge gateway", "Hardware", 89_900),
    ("Sensor kit", "Hardware", 24_900),
    ("Replacement sensor", "Hardware", 4_900),
)
_PAYMENT_METHODS = ("bank transfer", "card", "card", "ach")
_TICKET_CATEGORIES = ("billing", "bug", "how-to", "feature request", "outage")
_TICKET_PRIORITIES = ("low", "normal", "normal", "high", "urgent")

Row = tuple[object, ...]


def sample_tables() -> dict[str, tuple[Row, ...]]:
    """Return every sample row by table, in insertion order."""

    return _Generator(random.Random(_SEED)).tables()


def build_sample_database(path: Path) -> Path:
    """Write the sample database to ``path``, replacing any file there atomically."""

    tables = sample_tables()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(_SCHEMA)
            for table in SAMPLE_TABLES:
                rows = tables[table]
                placeholders = ", ".join("?" * len(rows[0]))
                connection.executemany(
                    f"INSERT INTO {table} VALUES ({placeholders})", rows
                )
            connection.commit()
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise RuntimeError("sample data violates its own foreign keys")
        finally:
            connection.close()
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _money(cents: int) -> float:
    return cents / 100


def _month_starts(first: date, last: date) -> list[date]:
    months = []
    current = first.replace(day=1)
    while current <= last:
        months.append(current)
        current = (current + timedelta(days=32)).replace(day=1)
    return months


def _days_in_month(month: date) -> int:
    return ((month + timedelta(days=32)).replace(day=1) - month).days


class _Generator:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def _date_between(self, first: date, last: date) -> date:
        return first + timedelta(days=self._rng.randint(0, (last - first).days))

    def tables(self) -> dict[str, tuple[Row, ...]]:
        rng = self._rng
        customers = self._customers()
        duplicates = rng.sample(range(1, _CUSTOMER_COUNT + 1), _DUPLICATE_COUNT)
        for original_id in duplicates:
            customers.append(self._duplicate(customers[original_id - 1], customers))
        late_payers = set(
            rng.sample(
                [
                    customer_id
                    for customer_id in range(1, _CUSTOMER_COUNT + 1)
                    if customer_id not in duplicates
                ],
                _LATE_PAYER_COUNT,
            )
        )
        accounts = self._accounts(customers)
        products = [
            (
                index,
                f"SKU-{1000 + index * 7}",
                name,
                category,
                _money(price),
            )
            for index, (name, category, price) in enumerate(_PRODUCTS, start=1)
        ]
        region_by_customer = {row[0]: row[3] for row in customers}
        orders, items = self._orders(accounts, region_by_customer)
        customer_by_account = {row[0]: row[1] for row in accounts}
        invoices, payments = self._invoices(orders, customer_by_account, late_payers)
        tickets = self._tickets(customers, late_payers)
        info = (
            ("as_of", SAMPLE_AS_OF.isoformat()),
            ("currency", "USD"),
            (
                "description",
                "Fictional B2B software and services business, "
                f"{_HISTORY_START.isoformat()} to {SAMPLE_AS_OF.isoformat()}.",
            ),
        )
        return {
            "dataset_info": info,
            "customers": tuple(customers),
            "accounts": tuple(accounts),
            "products": tuple(products),
            "orders": tuple(orders),
            "order_items": tuple(items),
            "invoices": tuple(invoices),
            "payments": tuple(payments),
            "support_tickets": tuple(tickets),
        }

    def _customers(self) -> list[Row]:
        rng = self._rng
        names = rng.sample(
            [(stem, kind) for stem in _NAME_STEMS for kind in _NAME_KINDS],
            _CUSTOMER_COUNT,
        )
        customers: list[Row] = []
        for customer_id, (stem, kind) in enumerate(names, start=1):
            suffix = rng.choice(_NAME_SUFFIXES)
            domain = f"{stem}{kind}".lower()
            customers.append(
                (
                    customer_id,
                    f"{stem} {kind} {suffix}",
                    f"billing@{domain}.example",
                    rng.choices(_REGIONS, weights=_REGION_WEIGHTS)[0],
                    rng.choice(_INDUSTRIES),
                    self._date_between(_HISTORY_START, date(2025, 6, 30)).isoformat(),
                )
            )
        return customers

    def _duplicate(self, original: Row, customers: list[Row]) -> Row:
        name = str(original[1])
        variants = (
            name.upper(),
            name.rsplit(" ", 1)[0],
            f"{name}.",
            name.replace(" ", "  ", 1),
        )
        domain = str(original[2]).split("@", 1)[1]
        created = date.fromisoformat(str(original[5])) + timedelta(
            days=self._rng.randint(60, 400)
        )
        return (
            len(customers) + 1,
            self._rng.choice(variants),
            f"accounts@{domain}",
            original[3],
            original[4],
            min(created, date(2025, 9, 30)).isoformat(),
        )

    def _accounts(self, customers: list[Row]) -> list[Row]:
        rng = self._rng
        accounts: list[Row] = []
        for customer in customers:
            count = 2 if rng.random() < 0.15 else 1
            for _ in range(count):
                created = date.fromisoformat(str(customer[5]))
                opened = self._date_between(created, created + timedelta(days=45))
                opened = min(opened, SAMPLE_AS_OF - timedelta(days=90))
                closed: date | None = None
                if rng.random() < 0.12:
                    closed = self._date_between(
                        opened + timedelta(days=90), SAMPLE_AS_OF
                    )
                accounts.append(
                    (
                        len(accounts) + 1,
                        customer[0],
                        rng.choices(_PLANS, weights=_PLAN_WEIGHTS)[0],
                        "active" if closed is None else "churned",
                        opened.isoformat(),
                        None if closed is None else closed.isoformat(),
                    )
                )
        return accounts

    def _orders(
        self,
        accounts: list[Row],
        region_by_customer: dict[object, object],
    ) -> tuple[list[Row], list[Row]]:
        rng = self._rng
        orders: list[Row] = []
        items: list[Row] = []
        for account in accounts:
            account_id, customer_id, plan = account[0], account[1], str(account[2])
            opened = date.fromisoformat(str(account[4]))
            last = (
                SAMPLE_AS_OF
                if account[5] is None
                else date.fromisoformat(str(account[5]))
            )
            # Some accounts still marked active stopped ordering months ago.
            stale_by = SAMPLE_AS_OF - timedelta(days=150)
            if (
                account[5] is None
                and opened + timedelta(days=60) < stale_by
                and rng.random() < 0.1
            ):
                last = self._date_between(opened + timedelta(days=60), stale_by)
            in_dip_region = region_by_customer[customer_id] == _DIP_REGION
            for month in _month_starts(opened, last):
                count = rng.choices(range(4), weights=_ORDER_WEIGHTS[plan])[0]
                for _ in range(count):
                    ordered = month + timedelta(
                        days=rng.randint(0, _days_in_month(month) - 1)
                    )
                    skipped = (
                        in_dip_region
                        and (month.year, month.month) in _DIP_MONTHS
                        and rng.random() < _DIP_SKIP_PROBABILITY
                    )
                    if ordered < opened or ordered > last or skipped:
                        continue
                    order_id = len(orders) + 1
                    total = 0
                    for product_index in sorted(
                        rng.sample(range(len(_PRODUCTS)), rng.randint(1, 3))
                    ):
                        quantity = rng.randint(1, _MAX_QUANTITY[plan])
                        price = _PRODUCTS[product_index][2]
                        if _PRODUCTS[product_index][1] == "Services":
                            quantity = 1
                        total += quantity * price
                        items.append(
                            (
                                len(items) + 1,
                                order_id,
                                product_index + 1,
                                quantity,
                                _money(price),
                            )
                        )
                    draw = rng.random()
                    status = (
                        "cancelled"
                        if draw < 0.04
                        else "refunded" if draw < 0.06 else "completed"
                    )
                    orders.append(
                        (
                            order_id,
                            account_id,
                            ordered.isoformat(),
                            status,
                            _money(total),
                        )
                    )
        return orders, items

    def _invoices(
        self,
        orders: list[Row],
        customer_by_account: dict[object, object],
        late_payers: set[int],
    ) -> tuple[list[Row], list[Row]]:
        rng = self._rng
        invoices: list[Row] = []
        payments: list[Row] = []
        for order in orders:
            order_id, account_id, ordered_at, status, total = order
            assert isinstance(total, float)
            if status == "cancelled":
                continue
            issued = date.fromisoformat(str(ordered_at))
            due = issued + timedelta(days=30)
            invoice_id = len(invoices) + 1
            late = customer_by_account[account_id] in late_payers
            paid = issued + timedelta(
                days=rng.randint(45, 110) if late else rng.randint(3, 30)
            )
            method = rng.choice(_PAYMENT_METHODS)
            if paid > SAMPLE_AS_OF:
                invoice_status = "overdue" if due < SAMPLE_AS_OF else "open"
            else:
                invoice_status = "refunded" if status == "refunded" else "paid"
                payments.append(
                    (len(payments) + 1, invoice_id, paid.isoformat(), total, method)
                )
                if status == "refunded":
                    refunded = min(
                        paid + timedelta(days=rng.randint(5, 40)), SAMPLE_AS_OF
                    )
                    payments.append(
                        (
                            len(payments) + 1,
                            invoice_id,
                            refunded.isoformat(),
                            -total,
                            "refund",
                        )
                    )
            invoices.append(
                (
                    invoice_id,
                    order_id,
                    issued.isoformat(),
                    due.isoformat(),
                    total,
                    invoice_status,
                )
            )
        return invoices, payments

    def _tickets(self, customers: list[Row], late_payers: set[int]) -> list[Row]:
        rng = self._rng
        tickets: list[Row] = []
        for customer in customers:
            customer_id = customer[0]
            created = date.fromisoformat(str(customer[5]))
            count = rng.randint(0, 6) + (4 if customer_id in late_payers else 0)
            for _ in range(count):
                opened = self._date_between(created, SAMPLE_AS_OF)
                category = (
                    "billing"
                    if customer_id in late_payers and rng.random() < 0.6
                    else rng.choice(_TICKET_CATEGORIES)
                )
                closed = opened + timedelta(days=rng.randint(0, 21))
                still_open = closed > SAMPLE_AS_OF or rng.random() < 0.03
                tickets.append(
                    (
                        len(tickets) + 1,
                        customer_id,
                        opened.isoformat(),
                        None if still_open else closed.isoformat(),
                        rng.choice(_TICKET_PRIORITIES),
                        category,
                        "open" if still_open else "closed",
                    )
                )
        return tickets
