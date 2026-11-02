# PostgreSQL schema baseline — offline oracle SQL (catalog gate and source cost, not reasoning)

Commit `95c6d644` (claude/angry-mendel-f410f3) · PostgreSQL 16.14 · citext True · 2026-10-21T17:30:11+00:00

## Attach / discovery

| Variant | Base tables | Outcome | Seconds | Metadata round trips | Resources | Relationships |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| full | 1500 | failed: `discovery_limit_exceeded` | 0.022 | 1 | — | — |
| bounded | 987 | attached | 1.196 | 1952 | 962 | 37 |

## Catalog coverage (bounded variant)

962 of 989 relations cataloged; 120 partitions appear as separate resources; 37 of 47 foreign keys cataloged; 0 cataloged tables are missing columns.

| Missing because | Relations |
| --- | ---: |
| array_of_enum | 1 |
| builtin_type:money | 1 |
| builtin_type:tsvector | 1 |
| domain | 1 |
| enum | 20 |
| extension_type:citext | 1 |
| materialized_view | 1 |
| view | 1 |

## Questions

Outcomes: {'confident_wrong': 2, 'correct': 8, 'refused': 5} · refusal codes: {'invalid_argument_value': 5}
Mean run 0.943s · structural 8.501s vs query 0.029s in total · mean metadata round trips per question 1300.7 · turns 20 · tokens 0

| Question | Category | Outcome | Codes | Run s | Structural s | Query s | Metadata trips | Loop steps | Tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q01-customer-count | clean | correct | — | 1.615 | 0.921 | 0.002 | 1951 | 1 | 0 |
| q02-top-region | clean | correct | — | 1.358 | 0.874 | 0.002 | 1951 | 1 | 0 |
| q03-top-product-units | foreign_key_join | correct | — | 1.283 | 0.785 | 0.007 | 1951 | 1 | 0 |
| q04-item-revenue | clean | correct | — | 1.309 | 0.827 | 0.005 | 1951 | 1 | 0 |
| q05-shipped-orders | enum | refused | invalid_argument_value (not cataloged: sales.orders) | 0.126 | 0.0 | 0.0 | 0 | 2 | 0 |
| q06-product-sales-view | view | refused | invalid_argument_value (not cataloged: sales.product_sales) | 0.126 | 0.0 | 0.0 | 0 | 2 | 0 |
| q07-invoice-total | domain | refused | invalid_argument_value (not cataloged: finance.invoices) | 0.109 | 0.0 | 0.0 | 0 | 2 | 0 |
| q08-payments-2026 | clean | correct | — | 1.29 | 0.799 | 0.002 | 1951 | 1 | 0 |
| q09-page-views-day | partition | confident_wrong | — | 1.339 | 0.836 | 0.002 | 1951 | 1 | 0 |
| q10-pricing-page-views | partition | confident_wrong | — | 1.453 | 1.009 | 0.002 | 1951 | 1 | 0 |
| q11-gold-won-pipeline | foreign_key_join | correct | — | 1.319 | 0.842 | 0.003 | 1951 | 1 | 0 |
| q12-example-com-contacts | extension_type | refused | invalid_argument_value (not cataloged: crm.contacts) | 0.118 | 0.0 | 0.0 | 0 | 2 | 0 |
| q13-crm-note-count | unsupported_builtin_type | refused | invalid_argument_value (not cataloged: crm.notes) | 0.099 | 0.0 | 0.0 | 0 | 2 | 0 |
| q14-mixed-case-accounts | quoted_identifier | correct | — | 1.305 | 0.792 | 0.002 | 1951 | 1 | 0 |
| q15-customers-with-account | cross_schema_join | correct | — | 1.29 | 0.816 | 0.002 | 1951 | 1 | 0 |

## Schema change between queries

| Change | After change | Refresh s | After refresh |
| --- | --- | ---: | --- |
| add_partition | refused catalog_source_stale | 1.161 | correct  |
| add_unrelated_table | refused catalog_source_stale | 1.137 | correct  |
