CREATE ROLE daita_reader
    LOGIN
    PASSWORD 'daita_fixture_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT;

ALTER ROLE daita_reader SET default_transaction_read_only = on;

CREATE SCHEMA analytics;

CREATE TABLE analytics.regions (
    region_code text PRIMARY KEY,
    region_name text NOT NULL UNIQUE,
    currency_code text NOT NULL
);

CREATE TABLE analytics.customers (
    customer_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_name text NOT NULL,
    email text NOT NULL UNIQUE,
    region_code text NOT NULL REFERENCES analytics.regions(region_code),
    segment text NOT NULL CHECK (segment IN ('enterprise', 'mid-market', 'smb')),
    signed_up_at timestamptz NOT NULL,
    is_active boolean NOT NULL
);

CREATE TABLE analytics.products (
    product_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sku text NOT NULL UNIQUE,
    product_name text NOT NULL,
    category text NOT NULL CHECK (
        category IN ('hardware', 'software', 'services', 'accessories')
    ),
    unit_price numeric(12, 2) NOT NULL CHECK (unit_price > 0),
    unit_cost numeric(12, 2) NOT NULL CHECK (unit_cost > 0),
    is_active boolean NOT NULL,
    created_at timestamptz NOT NULL
);

CREATE TABLE analytics.orders (
    order_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id bigint NOT NULL REFERENCES analytics.customers(customer_id),
    ordered_at timestamptz NOT NULL,
    status text NOT NULL CHECK (
        status IN ('paid', 'pending', 'refunded', 'cancelled')
    ),
    sales_channel text NOT NULL CHECK (
        sales_channel IN ('direct', 'partner', 'web', 'marketplace')
    ),
    subtotal numeric(14, 2) NOT NULL DEFAULT 0 CHECK (subtotal >= 0),
    tax_amount numeric(14, 2) NOT NULL DEFAULT 0 CHECK (tax_amount >= 0),
    total_amount numeric(14, 2) NOT NULL DEFAULT 0 CHECK (total_amount >= 0)
);

CREATE TABLE analytics.order_items (
    order_item_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id bigint NOT NULL REFERENCES analytics.orders(order_id),
    product_id bigint NOT NULL REFERENCES analytics.products(product_id),
    quantity integer NOT NULL CHECK (quantity BETWEEN 1 AND 10),
    unit_price numeric(12, 2) NOT NULL CHECK (unit_price > 0),
    discount_percent numeric(5, 2) NOT NULL CHECK (
        discount_percent BETWEEN 0 AND 100
    ),
    line_total numeric(14, 2) GENERATED ALWAYS AS (
        round(quantity * unit_price * (1 - discount_percent / 100), 2)
    ) STORED
);

CREATE TABLE analytics.payments (
    payment_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id bigint NOT NULL UNIQUE REFERENCES analytics.orders(order_id),
    processed_at timestamptz NOT NULL,
    payment_method text NOT NULL CHECK (
        payment_method IN ('card', 'bank_transfer', 'invoice', 'wallet')
    ),
    payment_status text NOT NULL CHECK (
        payment_status IN ('captured', 'refunded')
    ),
    amount numeric(14, 2) NOT NULL CHECK (amount >= 0)
);

CREATE TABLE analytics.shipments (
    shipment_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id bigint NOT NULL UNIQUE REFERENCES analytics.orders(order_id),
    warehouse_code text NOT NULL CHECK (
        warehouse_code IN ('CHI', 'DAL', 'FRA', 'SIN')
    ),
    shipment_status text NOT NULL CHECK (
        shipment_status IN ('processing', 'in_transit', 'delivered')
    ),
    shipped_at timestamptz,
    delivered_at timestamptz,
    CHECK (delivered_at IS NULL OR shipped_at IS NOT NULL),
    CHECK (delivered_at IS NULL OR delivered_at >= shipped_at)
);

CREATE TABLE analytics.support_tickets (
    ticket_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id bigint NOT NULL REFERENCES analytics.customers(customer_id),
    order_id bigint REFERENCES analytics.orders(order_id),
    opened_at timestamptz NOT NULL,
    category text NOT NULL CHECK (
        category IN ('billing', 'delivery', 'product', 'account')
    ),
    priority text NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    ticket_status text NOT NULL CHECK (
        ticket_status IN ('open', 'waiting', 'resolved')
    ),
    satisfaction_score integer CHECK (satisfaction_score BETWEEN 1 AND 5)
);

CREATE INDEX customers_region_code_idx
    ON analytics.customers(region_code);
CREATE INDEX orders_customer_id_idx
    ON analytics.orders(customer_id);
CREATE INDEX orders_ordered_at_idx
    ON analytics.orders(ordered_at);
CREATE INDEX orders_status_idx
    ON analytics.orders(status);
CREATE INDEX order_items_order_id_idx
    ON analytics.order_items(order_id);
CREATE INDEX order_items_product_id_idx
    ON analytics.order_items(product_id);
CREATE INDEX payments_processed_at_idx
    ON analytics.payments(processed_at);
CREATE INDEX support_tickets_customer_id_idx
    ON analytics.support_tickets(customer_id);
CREATE INDEX support_tickets_opened_at_idx
    ON analytics.support_tickets(opened_at);

INSERT INTO analytics.regions (region_code, region_name, currency_code) VALUES
    ('AMER', 'Americas', 'USD'),
    ('EMEA', 'Europe, Middle East, and Africa', 'EUR'),
    ('APAC', 'Asia Pacific', 'SGD');

INSERT INTO analytics.customers (
    customer_name,
    email,
    region_code,
    segment,
    signed_up_at,
    is_active
)
SELECT
    'Customer ' || upper(substr(md5(random()::text || series::text), 1, 10)),
    'customer-' || series || '-' || substr(md5(random()::text), 1, 8)
        || '@example.test',
    (ARRAY['AMER', 'EMEA', 'APAC'])[((series - 1) % 3) + 1],
    (ARRAY['enterprise', 'mid-market', 'smb'])[1 + floor(random() * 3)::integer],
    timestamptz '2022-01-01 00:00:00+00' + random() * interval '4 years',
    random() >= 0.08
FROM generate_series(1, 1000) AS generated(series);

WITH generated_products AS MATERIALIZED (
    SELECT
        series,
        round((10 + random() * 990)::numeric, 2) AS price
    FROM generate_series(1, 250) AS generated(series)
)
INSERT INTO analytics.products (
    sku,
    product_name,
    category,
    unit_price,
    unit_cost,
    is_active,
    created_at
)
SELECT
    'SKU-' || lpad(series::text, 5, '0'),
    'Product ' || upper(substr(md5(random()::text || series::text), 1, 12)),
    (ARRAY['hardware', 'software', 'services', 'accessories'])[
        ((series - 1) % 4) + 1
    ],
    price,
    round((price * (0.25 + random() * 0.5))::numeric, 2),
    random() >= 0.05,
    timestamptz '2021-01-01 00:00:00+00' + random() * interval '5 years'
FROM generated_products;

INSERT INTO analytics.orders (
    customer_id,
    ordered_at,
    status,
    sales_channel
)
SELECT
    1 + floor(random() * 1000)::bigint,
    timestamptz '2024-01-01 00:00:00+00' + random() * interval '2 years',
    (ARRAY['paid', 'paid', 'paid', 'paid', 'pending', 'refunded', 'cancelled'])[
        ((series - 1) % 7) + 1
    ],
    (ARRAY['direct', 'partner', 'web', 'marketplace'])[
        1 + floor(random() * 4)::integer
    ]
FROM generate_series(1, 6000) AS generated(series);

WITH generated_items AS MATERIALIZED (
    SELECT
        orders.order_id,
        1 + floor(random() * 250)::bigint AS product_id,
        1 + floor(random() * 5)::integer AS quantity,
        (ARRAY[0, 0, 0, 5, 10, 15, 20])[1 + floor(random() * 7)::integer]
            AS discount_percent
    FROM analytics.orders
    CROSS JOIN LATERAL generate_series(
        1,
        1 + floor(random() * 4 + orders.order_id * 0)::integer
    ) AS item_number
)
INSERT INTO analytics.order_items (
    order_id,
    product_id,
    quantity,
    unit_price,
    discount_percent
)
SELECT
    generated_items.order_id,
    generated_items.product_id,
    generated_items.quantity,
    products.unit_price,
    generated_items.discount_percent
FROM generated_items
JOIN analytics.products
    ON products.product_id = generated_items.product_id;

WITH order_totals AS MATERIALIZED (
    SELECT
        order_id,
        round(sum(line_total), 2) AS subtotal,
        round((sum(line_total) * (0.04 + random() * 0.06))::numeric, 2)
            AS tax_amount
    FROM analytics.order_items
    GROUP BY order_id
)
UPDATE analytics.orders
SET
    subtotal = order_totals.subtotal,
    tax_amount = order_totals.tax_amount,
    total_amount = order_totals.subtotal + order_totals.tax_amount
FROM order_totals
WHERE orders.order_id = order_totals.order_id;

INSERT INTO analytics.payments (
    order_id,
    processed_at,
    payment_method,
    payment_status,
    amount
)
SELECT
    order_id,
    ordered_at + random() * interval '3 days',
    (ARRAY['card', 'bank_transfer', 'invoice', 'wallet'])[
        1 + floor(random() * 4)::integer
    ],
    CASE WHEN status = 'refunded' THEN 'refunded' ELSE 'captured' END,
    total_amount
FROM analytics.orders
WHERE status IN ('paid', 'refunded');

WITH generated_shipments AS MATERIALIZED (
    SELECT
        order_id,
        ordered_at,
        random() AS state_sample,
        random() AS shipping_delay,
        random() AS delivery_delay
    FROM analytics.orders
    WHERE status IN ('paid', 'refunded')
)
INSERT INTO analytics.shipments (
    order_id,
    warehouse_code,
    shipment_status,
    shipped_at,
    delivered_at
)
SELECT
    order_id,
    (ARRAY['CHI', 'DAL', 'FRA', 'SIN'])[1 + floor(random() * 4)::integer],
    CASE
        WHEN state_sample < 0.12 THEN 'processing'
        WHEN state_sample < 0.38 THEN 'in_transit'
        ELSE 'delivered'
    END,
    CASE
        WHEN state_sample < 0.12 THEN NULL
        ELSE ordered_at + (1 + shipping_delay * 4) * interval '1 day'
    END,
    CASE
        WHEN state_sample < 0.38 THEN NULL
        ELSE ordered_at + (5 + delivery_delay * 10) * interval '1 day'
    END
FROM generated_shipments;

WITH generated_tickets AS MATERIALIZED (
    SELECT
        1 + floor(random() * 6000)::bigint AS order_id,
        random() AS resolution_sample
    FROM generate_series(1, 800)
)
INSERT INTO analytics.support_tickets (
    customer_id,
    order_id,
    opened_at,
    category,
    priority,
    ticket_status,
    satisfaction_score
)
SELECT
    orders.customer_id,
    orders.order_id,
    orders.ordered_at + random() * interval '45 days',
    (ARRAY['billing', 'delivery', 'product', 'account'])[
        1 + floor(random() * 4)::integer
    ],
    (ARRAY['low', 'medium', 'high', 'urgent'])[
        1 + floor(random() * 4)::integer
    ],
    CASE
        WHEN resolution_sample < 0.2 THEN 'open'
        WHEN resolution_sample < 0.35 THEN 'waiting'
        ELSE 'resolved'
    END,
    CASE
        WHEN resolution_sample < 0.35 THEN NULL
        ELSE 1 + floor(random() * 5)::integer
    END
FROM generated_tickets
JOIN analytics.orders
    ON orders.order_id = generated_tickets.order_id;

ANALYZE analytics.regions;
ANALYZE analytics.customers;
ANALYZE analytics.products;
ANALYZE analytics.orders;
ANALYZE analytics.order_items;
ANALYZE analytics.payments;
ANALYZE analytics.shipments;
ANALYZE analytics.support_tickets;

-- Make database admission explicit for fixture roles instead of inheriting the
-- PostgreSQL default CONNECT/TEMP privileges from PUBLIC. The owner/superuser
-- remains available for external fixture setup and verification.
REVOKE ALL PRIVILEGES ON DATABASE daita_fixture FROM PUBLIC;
REVOKE ALL PRIVILEGES ON DATABASE postgres FROM PUBLIC;
REVOKE ALL PRIVILEGES ON DATABASE template1 FROM PUBLIC;

GRANT CONNECT ON DATABASE daita_fixture TO daita_reader;
GRANT USAGE ON SCHEMA analytics TO daita_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO daita_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
    GRANT SELECT ON TABLES TO daita_reader;

-- Phase 4 write certification is isolated from the read fixture. This role and
-- schema are disposable test infrastructure provisioned by PostgreSQL startup,
-- never by Daita.
CREATE ROLE daita_writer
    LOGIN
    PASSWORD 'daita_writer_fixture_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    NOREPLICATION
    NOBYPASSRLS;

CREATE SCHEMA write_canary;

CREATE TABLE write_canary.regions (
    region_code text PRIMARY KEY,
    name text NOT NULL UNIQUE
);

CREATE TABLE write_canary.accounts (
    account_id bigint PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('active', 'inactive', 'locked')),
    external_key text NOT NULL UNIQUE,
    region_code text NOT NULL REFERENCES write_canary.regions(region_code),
    note text,
    counter integer NOT NULL CHECK (counter >= 0),
    updated_at timestamptz NOT NULL
);

CREATE TABLE write_canary.permission_denied (
    account_id bigint PRIMARY KEY,
    status text NOT NULL
);

CREATE TABLE write_canary.no_primary_key (
    account_id bigint NOT NULL,
    status text NOT NULL
);

CREATE TABLE write_canary.composite_primary_key (
    tenant_id bigint NOT NULL,
    account_id bigint NOT NULL,
    status text NOT NULL,
    PRIMARY KEY (tenant_id, account_id)
);

CREATE TABLE write_canary.rls_accounts (
    account_id bigint PRIMARY KEY,
    status text NOT NULL
);
ALTER TABLE write_canary.rls_accounts ENABLE ROW LEVEL SECURITY;

CREATE TABLE write_canary.trigger_accounts (
    account_id bigint PRIMARY KEY,
    status text NOT NULL
);

CREATE FUNCTION write_canary.reject_trigger_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'fixture trigger must never execute';
END;
$$;

CREATE TRIGGER reject_trigger_update
BEFORE UPDATE ON write_canary.trigger_accounts
FOR EACH ROW EXECUTE FUNCTION write_canary.reject_trigger_update();

INSERT INTO write_canary.regions (region_code, name) VALUES
    ('NA', 'North America'),
    ('EU', 'Europe');

INSERT INTO write_canary.accounts (
    account_id,
    status,
    external_key,
    region_code,
    note,
    counter,
    updated_at
) VALUES
    (42, 'active', 'canary-42', 'NA', 'phase-4 canary', 0,
        timestamptz '2026-08-10 00:00:00+00'),
    (43, 'inactive', 'canary-43', 'EU', 'constraint peer', 1,
        timestamptz '2026-08-10 00:00:00+00');

INSERT INTO write_canary.permission_denied VALUES (42, 'active');
INSERT INTO write_canary.no_primary_key VALUES (42, 'active');
INSERT INTO write_canary.composite_primary_key VALUES (1, 42, 'active');
INSERT INTO write_canary.rls_accounts VALUES (42, 'active');
INSERT INTO write_canary.trigger_accounts VALUES (42, 'active');

GRANT CONNECT ON DATABASE daita_fixture TO daita_writer;
GRANT USAGE ON SCHEMA write_canary TO daita_writer;
GRANT SELECT ON write_canary.regions TO daita_writer;
GRANT SELECT ON write_canary.accounts TO daita_writer;
GRANT UPDATE (
    status,
    external_key,
    region_code,
    note,
    counter,
    updated_at
) ON write_canary.accounts TO daita_writer;
GRANT SELECT ON write_canary.permission_denied TO daita_writer;
GRANT SELECT, UPDATE (status) ON write_canary.no_primary_key TO daita_writer;
GRANT SELECT, UPDATE (status) ON write_canary.composite_primary_key TO daita_writer;
GRANT SELECT, UPDATE (status) ON write_canary.rls_accounts TO daita_writer;
GRANT SELECT, UPDATE (status) ON write_canary.trigger_accounts TO daita_writer;
