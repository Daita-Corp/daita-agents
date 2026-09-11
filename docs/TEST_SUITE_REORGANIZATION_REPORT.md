# Test suite reorganization report

This report records the September 11, 2026 implementation of the test-suite
restructuring plan. Production code under `src/daita` was not changed. The
detailed, case-level review artifact is
[`TEST_SUITE_AUDIT_MANIFEST.json`](TEST_SUITE_AUDIT_MANIFEST.json).

## Result

Tests now live with their current production owner. Multi-owner public journeys
are in `tests/acceptance`, structural ownership checks are in
`tests/architecture`, real external execution is in `tests/live`, extended
offline soaks are in `tests/slow`, and reusable non-test code is in
`tests/support` or an owner-local support module. Diagnostic and packaging
scripts are importable module entry points. There are no `test_*.py` modules at
the root of `tests`.

Pytest uses `pythonpath = ["src", "."]`, strict registered markers, and default
directory exclusions for `tests/live` and `tests/slow`. CI uses the explicit
deterministic filter and retains its Python 3.11/3.12 matrices, candidate-wheel
reuse, and native installer jobs. Contributor guidance, the PR checklist,
fixture guides, examples, release instructions, and active script references
use the new paths.

## Inventory reconciliation

The baseline was captured before editing with all known live and soak gates
removed from the child environment. Static source counts and parameter-expanded
pytest cases are deliberately reported separately.

| Inventory | Baseline | Final | Reconciliation |
| --- | ---: | ---: | --- |
| Test modules | 141 | 184 | Owner splits increased cohesion; no root test modules remain |
| Source test definitions | 1,622 | 1,613 | -9 deletions, -2 merges, +2 net lazy-import replacements |
| Expanded collected cases | 2,773 | 2,759 | -14 deleted cases, -2 merges, +2 net replacement cases |
| Default deterministic cases | n/a under the old layout | 2,627 | All passed |
| Explicit live cases | 96 | 130 | External cases formerly at the root moved into `tests/live` |
| Explicit slow cases | 0 | 2 | The two extended concurrency cases moved into `tests/slow` |

The mechanical-move gate preserved all 2,773 expanded cases before quality
edits: normalized old-to-new node-ID comparison had zero missing or unexpected
cases. The final manifest contains each baseline ID exactly once. It references
every final ID; 2,757 final IDs have one predecessor and exactly two documented
merge survivors have two predecessors.

Source-definition decisions are: 1,567 keep, 31 reclassify, 13 rewrite, two
merge, and nine delete. Expanded-case decisions are: 2,678 keep, 66 reclassify,
13 rewrite, two merge, and 14 delete. The larger delete case count comes from
one six-case parameter family.

Marker reconciliation is exact at the case level. The final expanded counts are
184 `acceptance`, 134 `integration`, 62 `requires_db`, 82 `requires_llm`, one
`requires_network`, two `slow`, seven `unit`, 1,867 `asyncio`, 1,432
`parametrize`, and 135 `skipif`. The newly explicit `requires_network` and
`slow` classifications describe execution requirements that were previously
encoded only by location or local gates.

## Deletions and merges

Every intentional deletion follows. None concealed a failure or removed a
unique current contract.

| Removed definition | Reason and retained coverage |
| --- | --- |
| `test_survivor_docs_and_examples_describe_only_the_mvp` | Deleted a broad documentation-prose and historical-word oracle. Executable architecture ownership and public-surface checks remain in `tests/architecture/test_boundaries.py`. |
| `test_future_memory_skill_and_observation_surfaces_are_slim` | Deleted future-feature word bans that could pass or fail on spelling. Concrete memory, skill, observation, and architecture ownership tests remain. |
| `test_phase_c_removes_legacy_permission_runtime_and_unreleased_history` | Deleted repository-wide legacy-word scanning. Current permission owners, codecs, persistence, and runtime enforcement remain directly tested. |
| `test_loop_has_no_cross_cutting_lifecycle_responsibilities` | Deleted broad source-word bans duplicated by retained AST dependency and component-owner checks. |
| `test_hosted_download_returns_authenticated_handle_and_never_server_local_path` | Deleted a placeholder for a hosted-download surface that does not ship. Current artifact identity, read, save, and delivery boundaries remain tested. |
| `test_architecture_guide_documents_explicit_inactive_review_lifecycle` | Deleted documentation wording as an oracle. Default-disabled review, proposal persistence, explicit acceptance, and UI controls remain executable tests. |
| `test_managed_installer_documentation_describes_the_gated_release_pipeline` | Deleted a tautological documentation assertion. Release workflow structure and packaging entry points remain executable contracts; native release jobs were preserved. |
| `test_unsupported_workflow_has_no_invented_tool_or_terminal_fallback` | Deleted a canned model response that predetermined the refusal. Tool projection, failed dispatch, unsupported capability, and terminal-result behavior remain independently covered. |
| `test_live_fixtures_release_provider_and_open_agent_on_failure` (six cases) | Deleted a dynamic import that invoked other test functions and asserted only mocked harness choreography. Provider ownership, failed-open cleanup, cancellation, borrowed-client ownership, SDK transport release, and live fixture cleanup remain in their owner suites. |

The two duplicate definitions were merged only after comparing their scenarios
and oracles:

| Merged definition | Retained coverage |
| --- | --- |
| `test_sqlite_groups_runs_with_only_documented_columns_and_index` | `tests/storage/test_schema.py::test_current_state_schema_and_conversation_index_are_exact` (actual SQLite tables, columns, uniqueness, and index columns) |
| `test_conversation_grouping_adds_no_table_package_or_runtime` | `tests/architecture/test_boundaries.py::test_conversations_add_grouping_without_a_runtime_or_history_system`, with exact schema absence retained by the storage schema test |

## Important replacements and reclassifications

- The false source-order lazy-import check became three discriminating tests:
  fresh-process import isolation, an in-memory eager-import fault that proves the
  guard fails, and normalized missing-Textual dependency behavior.
- Architecture checks based on raw statement counts, private-local names, or
  milestone prose became focused AST ownership, dependency, registry, schema,
  and public-boundary assertions. Obsolete word bans were removed.
- Packaging documentation-string tests became a local Markdown link resolver
  and importable module-entry-point checks. These do not claim installed-wheel
  validation; CI retains that responsibility.
- The PostgreSQL fixture prose assertion became executable topology and
  least-privilege checks. Real PostgreSQL read/write cases moved to
  `tests/live/data` with their database and model requirements intact.
- Scripted-provider test names and assertions now claim deterministic dispatch,
  result propagation, or grounding inputs only; they do not claim model
  reasoning quality.
- Offline provider-failure, benchmark-harness, scheduled-MCP, and local text-edit
  qualification moved out of the old live hierarchy. Their reusable runners are
  support code, and the live wrappers remain independently gated and marked.
- Shared constructors, fakes, paths, transaction doubles, and live fixtures were
  extracted without introducing a second runtime. Test and support modules no
  longer import another `test_*.py` module or `conftest.py`.

The manifest records the exact destination, contract, boundary, scenario,
oracle, before/after markers, and evidence for every original definition and
expanded case, including all 31 source-level reclassifications.

## Protected contract review

| Protected boundary | Representative retained suites | Result |
| --- | --- | --- |
| Permission and frozen scope | `tests/context/test_source_scope.py`, `tests/security`, `tests/data/writes/test_public_writes.py`, `tests/mcp/test_runtime.py` | Preserved |
| Token, cost, step, and job limits | `tests/llm/test_budget_admission.py`, `tests/capabilities/test_runtime_bounds.py`, `tests/jobs/test_limits.py` | Preserved |
| Cancellation and cleanup | `tests/llm/test_call_lifecycle.py`, `tests/jobs/test_lifecycle.py`, `tests/workspace/test_text_edit.py` | Preserved |
| Rollback and drift rejection | `tests/data/writes/test_update_runtime.py`, `tests/data/writes/test_upsert.py`, `tests/live/data/test_write_release.py` | Preserved; real database execution remains live-only |
| Receipt certainty and recovery | `tests/capabilities/test_effect_receipts.py`, `tests/capabilities/test_effect_runtime.py`, `tests/capabilities/test_receipt_recovery_controls.py` | Preserved |
| Persistence and restart | `tests/storage`, `tests/jobs`, `tests/routines/test_storage.py`, permission persistence suites | Preserved |
| Exactly-once/no replay | effect runtime, provider-failure-after-admission, job and routine supervisor suites | Preserved |
| Artifact lineage and publication | `tests/artifacts`, `tests/workspace/test_text_edit.py`, scheduled delivery acceptance | Preserved |

No deterministic protected-contract gap was found. External PostgreSQL, paid
model, and remote MCP behavior remains represented by explicitly gated live
tests and was not treated as validated by offline fakes.

## Verification

The pre-edit Python 3.11 baseline collected 2,773 cases. Its explicit old-layout
deterministic selection reached 2,611 passes, three skips, 129 deselections, and
30 setup errors, all in local HTTP progress fixtures because the execution
sandbox denied loopback binding. The same local transport family passed when
the final offline suite was allowed to bind loopback; the baseline errors were
not removed or skipped.

Final results:

| Command or check | Result |
| --- | --- |
| `.venv/bin/python -m pytest` | 2,627 passed in 302.10s on Python 3.11 |
| Explicit deterministic CI marker filter | 2,627 passed in 300.30s |
| `.venv/bin/python -m pytest tests/llm -v` | 672 passed |
| Affected acceptance/capabilities/data-write/MCP/routine/TUI group | 670 passed |
| Affected data/job group | 429 passed |
| `.venv/bin/python -m pytest tests/architecture` | 65 passed |
| `python3.12 -m pytest tests/architecture` | 65 passed |
| Resident subprocess owner suite | 3 passed |
| Full explicit collection | 2,759 collected |
| Default collection | 2,627 collected; live and slow absent |
| Live collection with authorization flags unset | 130 collected; no execution |
| Slow selection with its enable flag unset | two skipped as gated |
| `python -m black --check src tests` | 465 files clean |
| `python -m ruff check --select I .` | passed |
| `python -m mypy src/daita tests` | 465 source files, no issues |
| Diagnostic and both packaging module `--help` checks | passed |
| Manifest-to-collection validator | zero missing, unknown, unreferenced, duplicate-original, or marker-mismatch cases |
| AST hygiene audit | zero cross-test/conftest imports, dynamic test imports, exact duplicate test bodies, missing package initializers, or unreferenced helper modules |
| `git diff --check` and obsolete active-path search | passed |

The local Python 3.12 interpreter passed the 65 architecture checks, but it is
not a fully provisioned project environment; full collection stopped on the
missing default dependency `textual`. No dependency was downloaded solely for
this reorganization. The unchanged CI matrix continues to install the project
and run Python 3.11 and 3.12. Candidate wheel lifecycles and the macOS/Linux
native installer matrix were not run locally; the module relocation was checked
with `--help`, and the existing CI jobs remain the installation evidence.

No paid model call, external database test, remote MCP call, install, publish,
release, credential load, or live gate was executed.
