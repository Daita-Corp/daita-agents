\set ON_ERROR_STOP on

SET client_min_messages = warning;
SET timezone = 'UTC';

CREATE ROLE daita_large_reader
    LOGIN
    PASSWORD 'daita_large_fixture_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT;

ALTER ROLE daita_large_reader SET default_transaction_read_only = on;

CREATE ROLE daita_large_writer
    LOGIN
    PASSWORD 'daita_large_writer_fixture_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    NOREPLICATION
    NOBYPASSRLS;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE SCHEMA core;
CREATE SCHEMA catalog;
CREATE SCHEMA sales;
CREATE SCHEMA billing;
CREATE SCHEMA support;
CREATE SCHEMA analytics;
CREATE SCHEMA archive;
CREATE SCHEMA staging;
CREATE SCHEMA private;

CREATE TABLE core.regions (
    region_code text PRIMARY KEY,
    region_name text NOT NULL UNIQUE,
    currency_code text NOT NULL,
    reporting_timezone text NOT NULL
);

CREATE TABLE core.countries (
    country_code text PRIMARY KEY,
    country_name text NOT NULL UNIQUE,
    region_code text NOT NULL REFERENCES core.regions(region_code)
);

CREATE TABLE core.organizations (
    organization_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_name text NOT NULL,
    region_code text NOT NULL REFERENCES core.regions(region_code),
    plan_name text NOT NULL CHECK (
        plan_name IN ('starter', 'growth', 'enterprise')
    ),
    created_at timestamptz NOT NULL,
    is_active boolean NOT NULL
);

CREATE TABLE core.customers (
    customer_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id bigint NOT NULL
        REFERENCES core.organizations(organization_id),
    country_code text NOT NULL REFERENCES core.countries(country_code),
    customer_name text NOT NULL,
    customer_key uuid NOT NULL UNIQUE,
    segment text NOT NULL CHECK (
        segment IN ('enterprise', 'mid-market', 'smb', 'consumer')
    ),
    signed_up_at timestamptz NOT NULL,
    is_active boolean NOT NULL
);

CREATE TABLE core.customer_contacts (
    contact_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id bigint NOT NULL REFERENCES core.customers(customer_id),
    email text NOT NULL UNIQUE,
    phone text,
    last_seen_ip inet,
    preferences jsonb NOT NULL
);

CREATE TABLE core.addresses (
    address_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id bigint NOT NULL REFERENCES core.customers(customer_id),
    country_code text NOT NULL REFERENCES core.countries(country_code),
    address_type text NOT NULL CHECK (
        address_type IN ('billing', 'shipping')
    ),
    city text NOT NULL,
    postal_code text NOT NULL,
    is_primary boolean NOT NULL,
    UNIQUE (customer_id, address_type)
);

CREATE TABLE core."CustomerSegments" (
    "SegmentID" integer PRIMARY KEY,
    "SegmentName" text NOT NULL UNIQUE,
    "ReportingLabel" text NOT NULL
);

CREATE TABLE catalog.categories (
    category_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    parent_category_id bigint REFERENCES catalog.categories(category_id),
    category_name text NOT NULL UNIQUE
);

CREATE TABLE catalog.suppliers (
    supplier_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    country_code text NOT NULL REFERENCES core.countries(country_code),
    supplier_name text NOT NULL,
    risk_tier text NOT NULL CHECK (risk_tier IN ('low', 'medium', 'high')),
    active_since date NOT NULL
);

CREATE TABLE catalog.products (
    product_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    category_id bigint NOT NULL REFERENCES catalog.categories(category_id),
    supplier_id bigint NOT NULL REFERENCES catalog.suppliers(supplier_id),
    sku text NOT NULL UNIQUE,
    product_name text NOT NULL,
    unit_price numeric(12, 2) NOT NULL CHECK (unit_price > 0),
    unit_cost numeric(12, 2) NOT NULL CHECK (unit_cost > 0),
    tags text[] NOT NULL,
    attributes jsonb NOT NULL,
    checksum bytea NOT NULL,
    created_at timestamptz NOT NULL,
    is_active boolean NOT NULL
);

CREATE TABLE catalog.warehouses (
    warehouse_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    warehouse_code text NOT NULL UNIQUE,
    region_code text NOT NULL REFERENCES core.regions(region_code),
    opened_on date NOT NULL
);

CREATE TABLE catalog.inventory (
    warehouse_id bigint NOT NULL
        REFERENCES catalog.warehouses(warehouse_id),
    product_id bigint NOT NULL REFERENCES catalog.products(product_id),
    quantity_on_hand integer NOT NULL CHECK (quantity_on_hand >= 0),
    reorder_point integer NOT NULL CHECK (reorder_point >= 0),
    last_counted_at timestamptz NOT NULL,
    PRIMARY KEY (warehouse_id, product_id)
);

CREATE TABLE catalog.price_books (
    price_book_code text PRIMARY KEY,
    price_book_name text NOT NULL UNIQUE,
    valid_from date NOT NULL,
    valid_to date
);

CREATE TABLE catalog.product_prices (
    price_book_code text NOT NULL
        REFERENCES catalog.price_books(price_book_code),
    product_id bigint NOT NULL REFERENCES catalog.products(product_id),
    list_price numeric(12, 2) NOT NULL CHECK (list_price > 0),
    PRIMARY KEY (price_book_code, product_id)
);

CREATE TYPE catalog.lifecycle_state AS ENUM (
    'planned',
    'active',
    'retired'
);

CREATE TABLE catalog.unsupported_type_probe (
    probe_id bigint PRIMARY KEY,
    state catalog.lifecycle_state NOT NULL
);

CREATE TABLE sales.sales_reps (
    sales_rep_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id bigint REFERENCES core.organizations(organization_id),
    region_code text NOT NULL REFERENCES core.regions(region_code),
    sales_rep_name text NOT NULL,
    hired_on date NOT NULL,
    quota_amount numeric(14, 2) NOT NULL CHECK (quota_amount > 0)
);

CREATE TABLE sales.channels (
    channel_code text PRIMARY KEY,
    channel_name text NOT NULL UNIQUE
);

CREATE TABLE sales.campaigns (
    campaign_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    channel_code text NOT NULL REFERENCES sales.channels(channel_code),
    campaign_name text NOT NULL,
    starts_on date NOT NULL,
    ends_on date NOT NULL,
    CHECK (ends_on >= starts_on)
);

CREATE TABLE sales.orders (
    order_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_key uuid NOT NULL UNIQUE,
    customer_id bigint NOT NULL REFERENCES core.customers(customer_id),
    sales_rep_id bigint NOT NULL REFERENCES sales.sales_reps(sales_rep_id),
    channel_code text NOT NULL REFERENCES sales.channels(channel_code),
    campaign_id bigint REFERENCES sales.campaigns(campaign_id),
    ordered_at timestamptz NOT NULL,
    status text NOT NULL CHECK (
        status IN ('paid', 'pending', 'refunded', 'cancelled')
    ),
    subtotal numeric(14, 2) NOT NULL DEFAULT 0 CHECK (subtotal >= 0),
    tax_amount numeric(14, 2) NOT NULL DEFAULT 0 CHECK (tax_amount >= 0),
    total_amount numeric(14, 2) NOT NULL DEFAULT 0 CHECK (total_amount >= 0)
);

CREATE TABLE sales.order_items (
    order_id bigint NOT NULL REFERENCES sales.orders(order_id),
    line_number integer NOT NULL CHECK (line_number BETWEEN 1 AND 20),
    product_id bigint NOT NULL REFERENCES catalog.products(product_id),
    quantity integer NOT NULL CHECK (quantity BETWEEN 1 AND 10),
    unit_price numeric(12, 2) NOT NULL CHECK (unit_price > 0),
    discount_percent numeric(5, 2) NOT NULL CHECK (
        discount_percent BETWEEN 0 AND 100
    ),
    line_total numeric(14, 2) GENERATED ALWAYS AS (
        round(quantity * unit_price * (1 - discount_percent / 100), 2)
    ) STORED,
    PRIMARY KEY (order_id, line_number)
);

CREATE TABLE sales.order_status_history (
    order_status_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id bigint NOT NULL REFERENCES sales.orders(order_id),
    status text NOT NULL CHECK (
        status IN ('created', 'paid', 'pending', 'refunded', 'cancelled')
    ),
    changed_at timestamptz NOT NULL
);

CREATE TABLE sales.returns (
    return_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id bigint NOT NULL,
    line_number integer NOT NULL,
    requested_at timestamptz NOT NULL,
    reason text NOT NULL CHECK (
        reason IN ('damaged', 'incorrect', 'late', 'unwanted')
    ),
    refund_amount numeric(14, 2) NOT NULL CHECK (refund_amount >= 0),
    FOREIGN KEY (order_id, line_number)
        REFERENCES sales.order_items(order_id, line_number)
);

CREATE TABLE billing.invoices (
    invoice_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id bigint NOT NULL UNIQUE REFERENCES sales.orders(order_id),
    customer_id bigint NOT NULL REFERENCES core.customers(customer_id),
    issued_at timestamptz NOT NULL,
    due_on date NOT NULL,
    status text NOT NULL CHECK (status IN ('paid', 'refunded')),
    subtotal numeric(14, 2) NOT NULL CHECK (subtotal >= 0),
    tax_amount numeric(14, 2) NOT NULL CHECK (tax_amount >= 0),
    total_amount numeric(14, 2) NOT NULL CHECK (total_amount >= 0)
);

CREATE TABLE billing.invoice_lines (
    invoice_id bigint NOT NULL REFERENCES billing.invoices(invoice_id),
    line_number integer NOT NULL,
    product_id bigint NOT NULL REFERENCES catalog.products(product_id),
    description text NOT NULL,
    quantity integer NOT NULL CHECK (quantity > 0),
    amount numeric(14, 2) NOT NULL CHECK (amount >= 0),
    PRIMARY KEY (invoice_id, line_number)
);

CREATE TABLE billing.payments (
    payment_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    invoice_id bigint NOT NULL UNIQUE REFERENCES billing.invoices(invoice_id),
    processed_at timestamptz NOT NULL,
    payment_method text NOT NULL CHECK (
        payment_method IN ('card', 'bank_transfer', 'invoice', 'wallet')
    ),
    payment_status text NOT NULL CHECK (
        payment_status IN ('captured', 'refunded')
    ),
    amount numeric(14, 2) NOT NULL CHECK (amount >= 0)
);

CREATE TABLE billing.refunds (
    refund_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    payment_id bigint NOT NULL REFERENCES billing.payments(payment_id),
    order_id bigint NOT NULL,
    line_number integer NOT NULL,
    refunded_at timestamptz NOT NULL,
    reason text NOT NULL,
    amount numeric(14, 2) NOT NULL CHECK (amount >= 0),
    FOREIGN KEY (order_id, line_number)
        REFERENCES sales.order_items(order_id, line_number)
);

CREATE TABLE support.agents (
    support_agent_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    region_code text NOT NULL REFERENCES core.regions(region_code),
    agent_name text NOT NULL,
    team_name text NOT NULL,
    hired_on date NOT NULL
);

CREATE TABLE support.tickets (
    ticket_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id bigint NOT NULL REFERENCES core.customers(customer_id),
    order_id bigint REFERENCES sales.orders(order_id),
    assigned_agent_id bigint REFERENCES support.agents(support_agent_id),
    opened_at timestamptz NOT NULL,
    closed_at timestamptz,
    category text NOT NULL CHECK (
        category IN ('billing', 'delivery', 'product', 'account')
    ),
    priority text NOT NULL CHECK (
        priority IN ('low', 'medium', 'high', 'urgent')
    ),
    ticket_status text NOT NULL CHECK (
        ticket_status IN ('open', 'waiting', 'resolved')
    ),
    satisfaction_score integer CHECK (satisfaction_score BETWEEN 1 AND 5),
    CHECK (closed_at IS NULL OR closed_at >= opened_at)
);

CREATE TABLE support.ticket_events (
    ticket_event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticket_id bigint NOT NULL REFERENCES support.tickets(ticket_id),
    actor_agent_id bigint REFERENCES support.agents(support_agent_id),
    event_type text NOT NULL CHECK (
        event_type IN ('opened', 'assigned', 'reply', 'resolved')
    ),
    occurred_at timestamptz NOT NULL,
    event_payload jsonb NOT NULL
);

CREATE TABLE analytics.daily_sales (
    sales_date date NOT NULL,
    region_code text NOT NULL REFERENCES core.regions(region_code),
    channel_code text NOT NULL REFERENCES sales.channels(channel_code),
    order_count bigint NOT NULL CHECK (order_count >= 0),
    gross_revenue numeric(16, 2) NOT NULL CHECK (gross_revenue >= 0),
    refunded_revenue numeric(16, 2) NOT NULL CHECK (refunded_revenue >= 0),
    PRIMARY KEY (sales_date, region_code, channel_code)
);

CREATE TABLE analytics.customer_health (
    customer_id bigint PRIMARY KEY REFERENCES core.customers(customer_id),
    lifetime_orders bigint NOT NULL CHECK (lifetime_orders >= 0),
    lifetime_revenue numeric(16, 2) NOT NULL CHECK (lifetime_revenue >= 0),
    open_ticket_count bigint NOT NULL CHECK (open_ticket_count >= 0),
    health_score numeric(6, 2) NOT NULL CHECK (
        health_score BETWEEN 0 AND 100
    ),
    calculated_at timestamptz NOT NULL
);

CREATE TABLE analytics.product_performance (
    product_id bigint PRIMARY KEY REFERENCES catalog.products(product_id),
    units_sold bigint NOT NULL CHECK (units_sold >= 0),
    gross_revenue numeric(16, 2) NOT NULL CHECK (gross_revenue >= 0),
    return_count bigint NOT NULL CHECK (return_count >= 0),
    calculated_at timestamptz NOT NULL
);

CREATE TABLE analytics.import_notes (
    import_batch text NOT NULL,
    note text NOT NULL,
    recorded_at timestamptz NOT NULL
);

CREATE TABLE archive.orders (
    order_id bigint PRIMARY KEY,
    customer_id bigint NOT NULL REFERENCES core.customers(customer_id),
    ordered_at timestamptz NOT NULL,
    status text NOT NULL,
    total_amount numeric(14, 2) NOT NULL
);

CREATE TABLE archive.order_items (
    order_id bigint NOT NULL REFERENCES archive.orders(order_id),
    line_number integer NOT NULL,
    product_id bigint NOT NULL REFERENCES catalog.products(product_id),
    quantity integer NOT NULL,
    line_total numeric(14, 2) NOT NULL,
    PRIMARY KEY (order_id, line_number)
);

CREATE TABLE private.payroll (
    employee_id bigint PRIMARY KEY,
    employee_name text NOT NULL,
    annual_salary numeric(14, 2) NOT NULL
);

INSERT INTO core.regions (
    region_code,
    region_name,
    currency_code,
    reporting_timezone
) VALUES
    ('AMER', 'Americas', 'USD', 'America/Chicago'),
    ('EMEA', 'Europe, Middle East, and Africa', 'EUR', 'Europe/Berlin'),
    ('APAC', 'Asia Pacific', 'SGD', 'Asia/Singapore'),
    ('LATAM', 'Latin America', 'BRL', 'America/Sao_Paulo'),
    ('CAN', 'Canada', 'CAD', 'America/Toronto');

INSERT INTO core.countries (country_code, country_name, region_code) VALUES
    ('US', 'United States', 'AMER'),
    ('DE', 'Germany', 'EMEA'),
    ('GB', 'United Kingdom', 'EMEA'),
    ('SG', 'Singapore', 'APAC'),
    ('JP', 'Japan', 'APAC'),
    ('BR', 'Brazil', 'LATAM'),
    ('MX', 'Mexico', 'LATAM'),
    ('CA', 'Canada', 'CAN'),
    ('AU', 'Australia', 'APAC'),
    ('FR', 'France', 'EMEA');

INSERT INTO core.organizations (
    organization_name,
    region_code,
    plan_name,
    created_at,
    is_active
)
SELECT
    'Organization ' || lpad(series::text, 4, '0'),
    (ARRAY['AMER', 'EMEA', 'APAC', 'LATAM', 'CAN'])[
        ((series - 1) % 5) + 1
    ],
    (ARRAY['starter', 'growth', 'enterprise'])[
        ((series - 1) % 3) + 1
    ],
    timestamptz '2020-01-01 00:00:00+00'
        + (series % 1825) * interval '1 day',
    series % 23 <> 0
FROM generate_series(1, 500) AS generated(series);

INSERT INTO core.customers (
    organization_id,
    country_code,
    customer_name,
    customer_key,
    segment,
    signed_up_at,
    is_active
)
SELECT
    ((series - 1) % 500) + 1,
    (ARRAY['US', 'DE', 'GB', 'SG', 'JP', 'BR', 'MX', 'CA', 'AU', 'FR'])[
        ((series - 1) % 10) + 1
    ],
    'Customer ' || upper(substr(md5(series::text), 1, 12)),
    (
        substr(md5('customer-' || series), 1, 8) || '-'
        || substr(md5('customer-' || series), 9, 4) || '-'
        || substr(md5('customer-' || series), 13, 4) || '-'
        || substr(md5('customer-' || series), 17, 4) || '-'
        || substr(md5('customer-' || series), 21, 12)
    )::uuid,
    (ARRAY['enterprise', 'mid-market', 'smb', 'consumer'])[
        ((series - 1) % 4) + 1
    ],
    timestamptz '2021-01-01 00:00:00+00'
        + (series % 1825) * interval '1 day',
    series % 19 <> 0
FROM generate_series(1, 10000) AS generated(series);

INSERT INTO core.customer_contacts (
    customer_id,
    email,
    phone,
    last_seen_ip,
    preferences
)
SELECT
    series,
    'customer-' || series || '@example.test',
    CASE
        WHEN series % 7 = 0 THEN NULL
        ELSE '+1-312-' || lpad((series % 10000)::text, 4, '0')
    END,
    (
        '10.'
        || (series % 250)
        || '.'
        || ((series * 7) % 250)
        || '.'
        || ((series * 13) % 250)
    )::inet,
    jsonb_build_object(
        'newsletter', series % 3 = 0,
        'locale', (ARRAY['en-US', 'de-DE', 'ja-JP'])[(series % 3) + 1]
    )
FROM generate_series(1, 10000) AS generated(series);

INSERT INTO core.addresses (
    customer_id,
    country_code,
    address_type,
    city,
    postal_code,
    is_primary
)
SELECT
    series,
    (ARRAY['US', 'DE', 'GB', 'SG', 'JP', 'BR', 'MX', 'CA', 'AU', 'FR'])[
        ((series - 1) % 10) + 1
    ],
    'billing',
    'City ' || ((series - 1) % 250 + 1),
    lpad((series % 100000)::text, 5, '0'),
    true
FROM generate_series(1, 10000) AS generated(series);

INSERT INTO core."CustomerSegments" (
    "SegmentID",
    "SegmentName",
    "ReportingLabel"
) VALUES
    (1, 'enterprise', 'Enterprise'),
    (2, 'mid-market', 'Mid-Market'),
    (3, 'smb', 'Small Business'),
    (4, 'consumer', 'Consumer');

INSERT INTO catalog.categories (category_name)
SELECT 'Category ' || lpad(series::text, 2, '0')
FROM generate_series(1, 50) AS generated(series);

UPDATE catalog.categories
SET parent_category_id = ((category_id - 2) % 10) + 1
WHERE category_id > 10;

INSERT INTO catalog.suppliers (
    country_code,
    supplier_name,
    risk_tier,
    active_since
)
SELECT
    (ARRAY['US', 'DE', 'GB', 'SG', 'JP', 'BR', 'MX', 'CA', 'AU', 'FR'])[
        ((series - 1) % 10) + 1
    ],
    'Supplier ' || lpad(series::text, 4, '0'),
    (ARRAY['low', 'medium', 'high'])[((series - 1) % 3) + 1],
    date '2018-01-01' + (series % 2000)
FROM generate_series(1, 500) AS generated(series);

INSERT INTO catalog.products (
    category_id,
    supplier_id,
    sku,
    product_name,
    unit_price,
    unit_cost,
    tags,
    attributes,
    checksum,
    created_at,
    is_active
)
SELECT
    ((series - 1) % 50) + 1,
    ((series * 17 - 1) % 500) + 1,
    'SKU-' || lpad(series::text, 7, '0'),
    'Product ' || upper(substr(md5('product-' || series), 1, 14)),
    round((10 + ((series * 37) % 99000)::numeric / 100), 2),
    round((5 + ((series * 19) % 50000)::numeric / 100), 2),
    ARRAY[
        (ARRAY['hardware', 'software', 'service', 'accessory'])[
            ((series - 1) % 4) + 1
        ],
        CASE WHEN series % 2 = 0 THEN 'featured' ELSE 'standard' END
    ],
    jsonb_build_object(
        'weight_kg', round((1 + (series % 250)::numeric / 10), 1),
        'warranty_months', 6 + (series % 31)
    ),
    decode(md5('product-' || series), 'hex'),
    timestamptz '2019-01-01 00:00:00+00'
        + (series % 2200) * interval '1 day',
    series % 29 <> 0
FROM generate_series(1, 5000) AS generated(series);

INSERT INTO catalog.warehouses (
    warehouse_code,
    region_code,
    opened_on
)
SELECT
    'WH-' || lpad(series::text, 3, '0'),
    (ARRAY['AMER', 'EMEA', 'APAC', 'LATAM', 'CAN'])[
        ((series - 1) % 5) + 1
    ],
    date '2015-01-01' + series * 90
FROM generate_series(1, 20) AS generated(series);

INSERT INTO catalog.inventory (
    warehouse_id,
    product_id,
    quantity_on_hand,
    reorder_point,
    last_counted_at
)
SELECT
    warehouse_id,
    product_id,
    (warehouse_id * 31 + product_id * 17) % 500,
    20 + (product_id % 50),
    timestamptz '2026-01-01 00:00:00+00'
        + ((warehouse_id + product_id) % 180) * interval '1 day'
FROM generate_series(1, 20) AS warehouses(warehouse_id)
CROSS JOIN generate_series(1, 5000) AS products(product_id);

INSERT INTO catalog.price_books (
    price_book_code,
    price_book_name,
    valid_from,
    valid_to
) VALUES
    ('STANDARD', 'Standard pricing', date '2024-01-01', NULL),
    ('PARTNER', 'Partner pricing', date '2024-01-01', NULL),
    ('ENTERPRISE', 'Enterprise pricing', date '2024-01-01', NULL);

INSERT INTO catalog.product_prices (
    price_book_code,
    product_id,
    list_price
)
SELECT
    price_book_code,
    product_id,
    round(
        products.unit_price
        * CASE price_book_code
            WHEN 'PARTNER' THEN 0.92
            WHEN 'ENTERPRISE' THEN 0.85
            ELSE 1
        END,
        2
    )
FROM catalog.products
CROSS JOIN (
    VALUES ('STANDARD'), ('PARTNER'), ('ENTERPRISE')
) AS books(price_book_code);

INSERT INTO catalog.unsupported_type_probe (probe_id, state) VALUES
    (1, 'active');

INSERT INTO sales.sales_reps (
    organization_id,
    region_code,
    sales_rep_name,
    hired_on,
    quota_amount
)
SELECT
    CASE WHEN series % 5 = 0 THEN NULL ELSE ((series - 1) % 500) + 1 END,
    (ARRAY['AMER', 'EMEA', 'APAC', 'LATAM', 'CAN'])[
        ((series - 1) % 5) + 1
    ],
    'Sales Rep ' || lpad(series::text, 3, '0'),
    date '2018-01-01' + (series % 2500),
    500000 + series * 2500
FROM generate_series(1, 250) AS generated(series);

INSERT INTO sales.channels (channel_code, channel_name) VALUES
    ('direct', 'Direct Sales'),
    ('partner', 'Partner'),
    ('web', 'Web'),
    ('marketplace', 'Marketplace');

INSERT INTO sales.campaigns (
    channel_code,
    campaign_name,
    starts_on,
    ends_on
)
SELECT
    (ARRAY['direct', 'partner', 'web', 'marketplace'])[
        ((series - 1) % 4) + 1
    ],
    'Campaign ' || lpad(series::text, 3, '0'),
    date '2024-01-01' + (series - 1) * 7,
    date '2024-01-01' + (series - 1) * 7 + 45
FROM generate_series(1, 100) AS generated(series);

INSERT INTO sales.orders (
    order_key,
    customer_id,
    sales_rep_id,
    channel_code,
    campaign_id,
    ordered_at,
    status
)
SELECT
    (
        substr(md5('order-' || series), 1, 8) || '-'
        || substr(md5('order-' || series), 9, 4) || '-'
        || substr(md5('order-' || series), 13, 4) || '-'
        || substr(md5('order-' || series), 17, 4) || '-'
        || substr(md5('order-' || series), 21, 12)
    )::uuid,
    ((series * 17 - 1) % 10000) + 1,
    ((series * 13 - 1) % 250) + 1,
    (ARRAY['direct', 'partner', 'web', 'marketplace'])[
        ((series - 1) % 4) + 1
    ],
    CASE WHEN series % 4 = 0 THEN NULL ELSE ((series - 1) % 100) + 1 END,
    timestamptz '2024-01-01 00:00:00+00'
        + (series % 730) * interval '1 day'
        + (series % 86400) * interval '1 second',
    CASE
        WHEN series % 10 < 7 THEN 'paid'
        WHEN series % 10 = 7 THEN 'pending'
        WHEN series % 10 = 8 THEN 'refunded'
        ELSE 'cancelled'
    END
FROM generate_series(1, 100000) AS generated(series);

INSERT INTO sales.order_items (
    order_id,
    line_number,
    product_id,
    quantity,
    unit_price,
    discount_percent
)
SELECT
    order_id,
    line_number,
    ((order_id * 29 + line_number * 101 - 1) % 5000) + 1,
    ((order_id + line_number) % 5) + 1,
    round(
        (
            10
            + (
                (
                    ((order_id * 29 + line_number * 101 - 1) % 5000) + 1
                )
                * 37
            ) % 99000::bigint
            / 100.0
        )::numeric,
        2
    ),
    ((order_id + line_number) % 5) * 5
FROM generate_series(1, 100000) AS orders(order_id)
CROSS JOIN generate_series(1, 3) AS lines(line_number);

WITH totals AS MATERIALIZED (
    SELECT
        order_id,
        round(sum(line_total), 2) AS subtotal
    FROM sales.order_items
    GROUP BY order_id
)
UPDATE sales.orders
SET
    subtotal = totals.subtotal,
    tax_amount = round(totals.subtotal * (0.04 + (orders.order_id % 6) / 100.0), 2),
    total_amount = totals.subtotal
        + round(totals.subtotal * (0.04 + (orders.order_id % 6) / 100.0), 2)
FROM totals
WHERE orders.order_id = totals.order_id;

INSERT INTO sales.order_status_history (
    order_id,
    status,
    changed_at
)
SELECT
    orders.order_id,
    CASE sequence.step
        WHEN 1 THEN 'created'
        ELSE orders.status
    END,
    orders.ordered_at + sequence.step * interval '1 hour'
FROM sales.orders
CROSS JOIN generate_series(1, 2) AS sequence(step);

INSERT INTO sales.returns (
    order_id,
    line_number,
    requested_at,
    reason,
    refund_amount
)
SELECT
    items.order_id,
    items.line_number,
    orders.ordered_at + interval '14 days',
    (ARRAY['damaged', 'incorrect', 'late', 'unwanted'])[
        ((items.order_id / 25 - 1) % 4) + 1
    ],
    items.line_total
FROM sales.order_items AS items
JOIN sales.orders AS orders ON orders.order_id = items.order_id
WHERE items.line_number = 1
  AND items.order_id % 25 = 0;

INSERT INTO billing.invoices (
    order_id,
    customer_id,
    issued_at,
    due_on,
    status,
    subtotal,
    tax_amount,
    total_amount
)
SELECT
    order_id,
    customer_id,
    ordered_at + interval '1 hour',
    (ordered_at + interval '31 days')::date,
    status,
    subtotal,
    tax_amount,
    total_amount
FROM sales.orders
WHERE status IN ('paid', 'refunded');

INSERT INTO billing.invoice_lines (
    invoice_id,
    line_number,
    product_id,
    description,
    quantity,
    amount
)
SELECT
    invoices.invoice_id,
    items.line_number,
    items.product_id,
    'Order ' || items.order_id || ' line ' || items.line_number,
    items.quantity,
    items.line_total
FROM billing.invoices AS invoices
JOIN sales.order_items AS items ON items.order_id = invoices.order_id;

INSERT INTO billing.payments (
    invoice_id,
    processed_at,
    payment_method,
    payment_status,
    amount
)
SELECT
    invoice_id,
    issued_at + interval '2 hours',
    (ARRAY['card', 'bank_transfer', 'invoice', 'wallet'])[
        ((invoice_id - 1) % 4) + 1
    ],
    CASE WHEN status = 'refunded' THEN 'refunded' ELSE 'captured' END,
    total_amount
FROM billing.invoices;

INSERT INTO billing.refunds (
    payment_id,
    order_id,
    line_number,
    refunded_at,
    reason,
    amount
)
SELECT
    payments.payment_id,
    invoices.order_id,
    1,
    payments.processed_at + interval '10 days',
    'Fixture refund for a returned or refunded order',
    items.line_total
FROM billing.payments AS payments
JOIN billing.invoices AS invoices
    ON invoices.invoice_id = payments.invoice_id
JOIN sales.order_items AS items
    ON items.order_id = invoices.order_id
   AND items.line_number = 1
WHERE payments.payment_status = 'refunded';

INSERT INTO support.agents (
    region_code,
    agent_name,
    team_name,
    hired_on
)
SELECT
    (ARRAY['AMER', 'EMEA', 'APAC', 'LATAM', 'CAN'])[
        ((series - 1) % 5) + 1
    ],
    'Support Agent ' || lpad(series::text, 3, '0'),
    (ARRAY['billing', 'delivery', 'product', 'account'])[
        ((series - 1) % 4) + 1
    ],
    date '2019-01-01' + (series % 2200)
FROM generate_series(1, 100) AS generated(series);

INSERT INTO support.tickets (
    customer_id,
    order_id,
    assigned_agent_id,
    opened_at,
    closed_at,
    category,
    priority,
    ticket_status,
    satisfaction_score
)
SELECT
    orders.customer_id,
    orders.order_id,
    ((series * 7 - 1) % 100) + 1,
    orders.ordered_at + (series % 30) * interval '1 day',
    CASE
        WHEN series % 5 < 3
        THEN orders.ordered_at + (series % 30 + 4) * interval '1 day'
        ELSE NULL
    END,
    (ARRAY['billing', 'delivery', 'product', 'account'])[
        ((series - 1) % 4) + 1
    ],
    (ARRAY['low', 'medium', 'high', 'urgent'])[
        ((series - 1) % 4) + 1
    ],
    CASE
        WHEN series % 5 < 3 THEN 'resolved'
        WHEN series % 5 = 3 THEN 'waiting'
        ELSE 'open'
    END,
    CASE WHEN series % 5 < 3 THEN (series % 5) + 1 ELSE NULL END
FROM generate_series(1, 20000) AS generated(series)
JOIN sales.orders AS orders
    ON orders.order_id = ((series * 31 - 1) % 100000) + 1;

INSERT INTO support.ticket_events (
    ticket_id,
    actor_agent_id,
    event_type,
    occurred_at,
    event_payload
)
SELECT
    tickets.ticket_id,
    CASE
        WHEN sequence.step = 1 THEN NULL
        ELSE ((tickets.ticket_id * 11 - 1) % 100) + 1
    END,
    CASE
        WHEN sequence.step = 1 THEN 'opened'
        WHEN tickets.ticket_status = 'resolved' THEN 'resolved'
        ELSE 'reply'
    END,
    tickets.opened_at + sequence.step * interval '2 hours',
    jsonb_build_object(
        'sequence', sequence.step,
        'source', 'fixture'
    )
FROM support.tickets
CROSS JOIN generate_series(1, 2) AS sequence(step);

INSERT INTO analytics.daily_sales (
    sales_date,
    region_code,
    channel_code,
    order_count,
    gross_revenue,
    refunded_revenue
)
SELECT
    orders.ordered_at::date,
    organizations.region_code,
    orders.channel_code,
    count(*),
    round(
        sum(CASE WHEN orders.status = 'paid' THEN orders.total_amount ELSE 0 END),
        2
    ),
    round(
        sum(
            CASE WHEN orders.status = 'refunded' THEN orders.total_amount ELSE 0 END
        ),
        2
    )
FROM sales.orders AS orders
JOIN core.customers AS customers
    ON customers.customer_id = orders.customer_id
JOIN core.organizations AS organizations
    ON organizations.organization_id = customers.organization_id
GROUP BY
    orders.ordered_at::date,
    organizations.region_code,
    orders.channel_code;

WITH order_totals AS MATERIALIZED (
    SELECT
        customer_id,
        count(*) AS lifetime_orders,
        round(
            sum(
                CASE WHEN status IN ('paid', 'refunded') THEN total_amount ELSE 0 END
            ),
            2
        ) AS lifetime_revenue
    FROM sales.orders
    GROUP BY customer_id
),
ticket_totals AS MATERIALIZED (
    SELECT
        customer_id,
        count(*) FILTER (WHERE ticket_status <> 'resolved') AS open_ticket_count
    FROM support.tickets
    GROUP BY customer_id
)
INSERT INTO analytics.customer_health (
    customer_id,
    lifetime_orders,
    lifetime_revenue,
    open_ticket_count,
    health_score,
    calculated_at
)
SELECT
    customers.customer_id,
    coalesce(order_totals.lifetime_orders, 0),
    coalesce(order_totals.lifetime_revenue, 0),
    coalesce(ticket_totals.open_ticket_count, 0),
    greatest(
        0,
        100
        - coalesce(ticket_totals.open_ticket_count, 0) * 7
        - CASE WHEN customers.is_active THEN 0 ELSE 20 END
    ),
    timestamptz '2026-07-01 00:00:00+00'
FROM core.customers AS customers
LEFT JOIN order_totals ON order_totals.customer_id = customers.customer_id
LEFT JOIN ticket_totals ON ticket_totals.customer_id = customers.customer_id;

WITH item_totals AS MATERIALIZED (
    SELECT
        product_id,
        sum(quantity) AS units_sold,
        round(sum(line_total), 2) AS gross_revenue
    FROM sales.order_items
    GROUP BY product_id
),
return_totals AS MATERIALIZED (
    SELECT
        items.product_id,
        count(*) AS return_count
    FROM sales.returns AS returns
    JOIN sales.order_items AS items
      ON items.order_id = returns.order_id
     AND items.line_number = returns.line_number
    GROUP BY items.product_id
)
INSERT INTO analytics.product_performance (
    product_id,
    units_sold,
    gross_revenue,
    return_count,
    calculated_at
)
SELECT
    products.product_id,
    coalesce(item_totals.units_sold, 0),
    coalesce(item_totals.gross_revenue, 0),
    coalesce(return_totals.return_count, 0),
    timestamptz '2026-07-01 00:00:00+00'
FROM catalog.products
LEFT JOIN item_totals ON item_totals.product_id = products.product_id
LEFT JOIN return_totals ON return_totals.product_id = products.product_id;

INSERT INTO analytics.import_notes (
    import_batch,
    note,
    recorded_at
) VALUES
    ('batch-001', 'Initial deterministic load', '2026-07-01 00:00:00+00'),
    ('batch-002', 'Validated cross-schema relationships', '2026-07-01 01:00:00+00'),
    ('batch-003', 'No primary key is intentional', '2026-07-01 02:00:00+00');

INSERT INTO archive.orders (
    order_id,
    customer_id,
    ordered_at,
    status,
    total_amount
)
SELECT
    order_id,
    customer_id,
    ordered_at - interval '2 years',
    status,
    total_amount
FROM sales.orders
WHERE order_id <= 20000;

INSERT INTO archive.order_items (
    order_id,
    line_number,
    product_id,
    quantity,
    line_total
)
SELECT
    items.order_id,
    items.line_number,
    items.product_id,
    items.quantity,
    items.line_total
FROM sales.order_items AS items
WHERE items.order_id <= 20000;

INSERT INTO private.payroll (
    employee_id,
    employee_name,
    annual_salary
) VALUES
    (1, 'Private Fixture Employee', 150000.00);

CREATE VIEW analytics.monthly_revenue AS
SELECT
    date_trunc('month', sales_date)::date AS sales_month,
    region_code,
    channel_code,
    sum(gross_revenue) AS gross_revenue,
    sum(refunded_revenue) AS refunded_revenue
FROM analytics.daily_sales
GROUP BY
    date_trunc('month', sales_date)::date,
    region_code,
    channel_code;

CREATE INDEX countries_region_code_idx
    ON core.countries(region_code);
CREATE INDEX organizations_region_code_idx
    ON core.organizations(region_code);
CREATE INDEX customers_organization_id_idx
    ON core.customers(organization_id);
CREATE INDEX customers_country_code_idx
    ON core.customers(country_code);
CREATE INDEX customer_contacts_customer_id_idx
    ON core.customer_contacts(customer_id);
CREATE INDEX addresses_customer_id_idx
    ON core.addresses(customer_id);
CREATE INDEX suppliers_country_code_idx
    ON catalog.suppliers(country_code);
CREATE INDEX products_category_id_idx
    ON catalog.products(category_id);
CREATE INDEX products_supplier_id_idx
    ON catalog.products(supplier_id);
CREATE INDEX inventory_product_id_idx
    ON catalog.inventory(product_id);
CREATE INDEX product_prices_product_id_idx
    ON catalog.product_prices(product_id);
CREATE INDEX sales_reps_region_code_idx
    ON sales.sales_reps(region_code);
CREATE INDEX campaigns_channel_code_idx
    ON sales.campaigns(channel_code);
CREATE INDEX orders_customer_id_idx
    ON sales.orders(customer_id);
CREATE INDEX orders_ordered_at_idx
    ON sales.orders(ordered_at);
CREATE INDEX orders_status_idx
    ON sales.orders(status);
CREATE INDEX orders_channel_code_idx
    ON sales.orders(channel_code);
CREATE INDEX order_items_product_id_idx
    ON sales.order_items(product_id);
CREATE INDEX order_status_history_order_id_idx
    ON sales.order_status_history(order_id);
CREATE INDEX returns_order_line_idx
    ON sales.returns(order_id, line_number);
CREATE INDEX invoices_customer_id_idx
    ON billing.invoices(customer_id);
CREATE INDEX invoices_status_idx
    ON billing.invoices(status);
CREATE INDEX invoice_lines_product_id_idx
    ON billing.invoice_lines(product_id);
CREATE INDEX refunds_payment_id_idx
    ON billing.refunds(payment_id);
CREATE INDEX refunds_order_line_idx
    ON billing.refunds(order_id, line_number);
CREATE INDEX tickets_customer_id_idx
    ON support.tickets(customer_id);
CREATE INDEX tickets_order_id_idx
    ON support.tickets(order_id);
CREATE INDEX tickets_status_idx
    ON support.tickets(ticket_status);
CREATE INDEX ticket_events_ticket_id_idx
    ON support.ticket_events(ticket_id);
CREATE INDEX archive_orders_customer_id_idx
    ON archive.orders(customer_id);
CREATE INDEX archive_order_items_product_id_idx
    ON archive.order_items(product_id);

ANALYZE;

REVOKE ALL PRIVILEGES ON DATABASE daita_large_fixture FROM PUBLIC;
REVOKE ALL PRIVILEGES ON DATABASE postgres FROM PUBLIC;
REVOKE ALL PRIVILEGES ON DATABASE template1 FROM PUBLIC;

GRANT CONNECT ON DATABASE daita_large_fixture TO daita_large_reader;
GRANT USAGE ON SCHEMA
    core,
    catalog,
    sales,
    billing,
    support,
    analytics,
    archive
TO daita_large_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA
    core,
    catalog,
    sales,
    billing,
    support,
    analytics,
    archive
TO daita_large_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA
    core,
    catalog,
    sales,
    billing,
    support,
    analytics,
    archive
GRANT SELECT ON TABLES TO daita_large_reader;

-- Keep the production-shape reader read-only. This separate disposable role
-- exists only for terminal single-selection and bulk update testing against
-- one existing large fixture table and one exact assignment column.
GRANT CONNECT ON DATABASE daita_large_fixture TO daita_large_writer;
GRANT USAGE ON SCHEMA support TO daita_large_writer;
GRANT SELECT ON support.tickets TO daita_large_writer;
GRANT UPDATE (priority) ON support.tickets TO daita_large_writer;

CREATE TABLE private.fixture_status (
    ready boolean PRIMARY KEY
);

INSERT INTO private.fixture_status (ready) VALUES (true);
