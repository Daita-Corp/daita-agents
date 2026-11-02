# PostgreSQL schema baseline — Daita vs Claude Code

Commit `2316c8cc` (HEAD) · PostgreSQL 16.14 · Claude Code None · 2026-11-02T14:46:42+00:00 · 987 tables in scope for both, read-only role · 3 repeat(s)

Limits per question: Daita 24 steps, 200000 tokens, 300.0 s; Claude Code $0.50.

| System | Model | Runs | Correct | Confident-wrong | No answer | Incomplete | Mean s | Turns | Tool calls | Tokens | Input tokens/turn | Cost USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| daita | anthropic:claude-sonnet-5 | 45 | 25 | 2 | 6 | 12 | 21.5 | 214 | 226 | 4213232 | 19420 | unknown |

| Question | Category | daita correct | daita outcomes |
| --- | --- | ---: | --- |
| q01-customer-count | clean | 3/3 | — |
| q02-top-region | clean | 3/3 | — |
| q03-top-product-units | foreign_key_join | 3/3 | — |
| q04-item-revenue | clean | 3/3 | — |
| q05-shipped-orders | enum | 0/3 | incomplete, incomplete, incomplete |
| q06-product-sales-view | view | 0/3 | no_answer, no_answer, no_answer |
| q07-invoice-total | domain | 0/3 | incomplete, incomplete, no_answer |
| q08-payments-2026 | clean | 3/3 | — |
| q09-page-views-day | partition | 3/3 | — |
| q10-pricing-page-views | partition | 0/3 | confident_wrong, confident_wrong, incomplete |
| q11-gold-won-pipeline | foreign_key_join | 3/3 | — |
| q12-example-com-contacts | extension_type | 0/3 | incomplete, incomplete, incomplete |
| q13-crm-note-count | unsupported_builtin_type | 0/3 | incomplete, no_answer, no_answer |
| q14-mixed-case-accounts | quoted_identifier | 3/3 | — |
| q15-customers-with-account | cross_schema_join | 1/3 | incomplete, incomplete |

Scoring reads the requested last line: correct when it holds the known value; no answer when the run declines or asks for clarification; incomplete when a limit or error stopped the run; otherwise confidently wrong. It is a keyword rule, not a human label; `answer_elsewhere` in the JSON flags answers outside the last line.
