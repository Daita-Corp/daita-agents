-- Deterministic sales data for the cross-source benchmark (PostgreSQL side).
-- Every run drops and recreates schema sales, so it only runs against the
-- dedicated daita_cross_source database. The harness runs each
-- blank-line-separated chunk as its own transaction; keep blocks free of blank
-- lines. No random(): every value is arithmetic on the series index.
--
-- Planted against the SQLite CRM and the folder files (see questions.json):
--   orders.customer_id runs 1..52 but the CRM holds customers 1..50 and 60;
--   invoices.account_code is 'ACC-001'..'ACC-050' where the CRM spells three
--   of them 'acc-013', 'ACC-022 ' and 'ACC-034 ';
--   order_items.product_id is an integer where the product file holds '001'..'023'.

DO $$
BEGIN
    IF current_database() <> 'daita_cross_source' THEN
        RAISE EXCEPTION 'refusing to seed database %', current_database();
    END IF;
END
$$;

DROP SCHEMA IF EXISTS sales CASCADE;

CREATE SCHEMA sales;

CREATE TABLE sales.orders (
    order_id integer PRIMARY KEY,
    customer_id integer NOT NULL,
    status text NOT NULL,
    total integer NOT NULL
);

INSERT INTO sales.orders
SELECT g,
       1 + (g * 7) % 52,
       CASE g % 5 WHEN 0 THEN 'cancelled' WHEN 1 THEN 'pending' ELSE 'shipped' END,
       10 + (g * 37) % 90
FROM generate_series(1, 3000) AS g;

-- 120,000 rows: more than the 100,000 rows all combined-query inputs may share,
-- so this table must be aggregated or filtered inside its input.
CREATE TABLE sales.order_items (
    item_id integer PRIMARY KEY,
    order_id integer NOT NULL REFERENCES sales.orders (order_id),
    product_id integer NOT NULL,
    quantity integer NOT NULL
);

INSERT INTO sales.order_items
SELECT g,
       1 + (g - 1) / 40,
       1 + (g * 3) % 20,
       1 + g % 5
FROM generate_series(1, 120000) AS g;

CREATE TABLE sales.invoices (
    invoice_id integer PRIMARY KEY,
    account_code text NOT NULL,
    amount integer NOT NULL
);

INSERT INTO sales.invoices
SELECT g,
       'ACC-' || lpad((1 + (g - 1) / 2)::text, 3, '0'),
       100 + (g * 173) % 900
FROM generate_series(1, 100) AS g;

ANALYZE;

-- Read-only login shared by Daita and Claude Code in the live comparison.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'daita_cross_source_reader') THEN
        CREATE ROLE daita_cross_source_reader LOGIN PASSWORD 'daita_cross_source_reader_password';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA sales TO daita_cross_source_reader;

GRANT SELECT ON ALL TABLES IN SCHEMA sales TO daita_cross_source_reader;
