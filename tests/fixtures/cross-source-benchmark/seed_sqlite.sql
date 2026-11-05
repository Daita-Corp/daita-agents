-- Deterministic CRM for the cross-source benchmark (SQLite side).
-- Customers 1..50 plus 60, who never ordered. Accounts 13, 22 and 34 spell
-- their code differently from the PostgreSQL invoices.
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    segment TEXT NOT NULL,
    country TEXT NOT NULL
);

WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 50)
INSERT INTO customers
SELECT i,
       printf('Customer %03d', i),
       CASE i % 4 WHEN 0 THEN 'enterprise' WHEN 1 THEN 'mid' ELSE 'smb' END,
       CASE i % 7 WHEN 0 THEN 'DE' WHEN 1 THEN 'DE' WHEN 2 THEN 'FR' WHEN 3 THEN 'US'
                  WHEN 4 THEN 'GB' WHEN 5 THEN 'ES' ELSE 'BR' END
FROM n;

INSERT INTO customers VALUES (60, 'Customer 060', 'smb', 'US');

CREATE TABLE accounts (
    account_code TEXT NOT NULL,
    customer_id INTEGER NOT NULL REFERENCES customers (id),
    plan TEXT NOT NULL
);

WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 50)
INSERT INTO accounts
SELECT CASE i WHEN 13 THEN 'acc-013' WHEN 22 THEN 'ACC-022 ' WHEN 34 THEN 'ACC-034 '
              ELSE printf('ACC-%03d', i) END,
       i,
       CASE i % 3 WHEN 0 THEN 'basic' WHEN 1 THEN 'pro' ELSE 'team' END
FROM n;
