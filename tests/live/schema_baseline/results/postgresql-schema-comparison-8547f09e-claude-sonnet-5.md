# PostgreSQL schema baseline — Daita vs Claude Code

Commit `8547f09e` (claude/angry-mendel-f410f3) · PostgreSQL 16.14 · Claude Code 2.1.306 (Claude Code) · 2026-10-21T20:35:09+00:00 · 987 tables in scope for both, read-only role · 3 repeat(s)

Limits per question: Daita 24 steps, 100000 tokens, 300.0 s; Claude Code $0.50.

| System | Model | Runs | Correct | Confident-wrong | No answer | Incomplete | Mean s | Turns | Tool calls | Tokens | Input tokens/turn | Cost USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| daita | anthropic:claude-sonnet-5 | 45 | 13 | 4 | 19 | 9 | 20.6 | 172 | 147 | 2643492 | 15043 | unknown |
| claude_code | claude-sonnet-5 | 45 | 45 | 0 | 0 | 0 | 9.7 | 208 | 163 | 3368196 | 16078 | 1.9054 |

| Question | Category | daita correct | daita outcomes | claude_code correct | claude_code outcomes |
| --- | --- | ---: | --- | ---: | --- |
| q01-customer-count | clean | 1/3 | incomplete, no_answer | 3/3 | — |
| q02-top-region | clean | 0/3 | no_answer, no_answer, no_answer | 3/3 | — |
| q03-top-product-units | foreign_key_join | 1/3 | incomplete, incomplete | 3/3 | — |
| q04-item-revenue | clean | 3/3 | — | 3/3 | — |
| q05-shipped-orders | enum | 0/3 | confident_wrong, confident_wrong, confident_wrong | 3/3 | — |
| q06-product-sales-view | view | 0/3 | no_answer, no_answer, no_answer | 3/3 | — |
| q07-invoice-total | domain | 0/3 | no_answer, no_answer, no_answer | 3/3 | — |
| q08-payments-2026 | clean | 3/3 | — | 3/3 | — |
| q09-page-views-day | partition | 2/3 | incomplete | 3/3 | — |
| q10-pricing-page-views | partition | 0/3 | confident_wrong, incomplete, no_answer | 3/3 | — |
| q11-gold-won-pipeline | foreign_key_join | 1/3 | no_answer, no_answer | 3/3 | — |
| q12-example-com-contacts | extension_type | 0/3 | incomplete, incomplete, incomplete | 3/3 | — |
| q13-crm-note-count | unsupported_builtin_type | 0/3 | no_answer, no_answer, no_answer | 3/3 | — |
| q14-mixed-case-accounts | quoted_identifier | 2/3 | incomplete | 3/3 | — |
| q15-customers-with-account | cross_schema_join | 0/3 | no_answer, no_answer, no_answer | 3/3 | — |

Scoring reads the requested last line: correct when it holds the known value; no answer when the run declines or asks for clarification; incomplete when a limit or error stopped the run; otherwise confidently wrong. It is a keyword rule, not a human label; `answer_elsewhere` in the JSON flags answers outside the last line.
