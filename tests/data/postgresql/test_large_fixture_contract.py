"""Component-owned tests split from ``test_postgres_large_fixture.py``."""

from __future__ import annotations

from tests.data.postgresql._large_fixture_support import (
    ATTACHED_SCHEMAS,
    FIXTURE,
)


def test_large_fixture_defines_topology_and_least_privilege():
    compose = (FIXTURE / "compose.yaml").read_text(encoding="utf-8")
    init = (FIXTURE / "init.sql").read_text(encoding="utf-8")

    assert "name: daita-postgres-large-fixture" in compose
    assert "${DAITA_LARGE_POSTGRES_PORT:-55433}" in compose
    assert "private.fixture_status" in compose
    for schema in (*ATTACHED_SCHEMAS, "private", "staging"):
        assert f"CREATE SCHEMA {schema};" in init
    assert "CREATE TABLE sales.orders" in init
    assert "CREATE TABLE archive.orders" in init
    assert "REFERENCES sales.orders(order_id)" in init
    assert "REFERENCES core.customers(customer_id)" in init
    assert "CREATE TYPE catalog.lifecycle_state AS ENUM" in init
    assert "CREATE VIEW analytics.monthly_revenue" in init
    assert "daita_large_reader" in init
    assert "CREATE ROLE daita_large_writer" in init
    assert "NOBYPASSRLS" in init
    assert "REVOKE ALL PRIVILEGES ON DATABASE daita_large_fixture FROM PUBLIC" in init
    assert "GRANT SELECT ON support.tickets TO daita_large_writer" in init
    assert "GRANT UPDATE (priority) ON support.tickets TO daita_large_writer" in init
