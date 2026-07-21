# Daita v1 to v2 parity and disposition matrix

This is the Phase 0 (`P0-04`) accounting record for behavior and public
surface parity. The v1 oracle is commit
`b87df31873d33fffbf50498f5dc4d8892115e8f8`; v1 source is a read-only
reference and is never a v2 dependency. The governing requirements are
Sections 4 and 16--19 of `docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md`.

No row authorizes Phase 10, removal of root `daita/`, or a v1 compatibility
fallback. A changed signature or replacement surface must be documented in
the Phase 9 migration/release notes even when the underlying behavior is
retained.

## Milestone and disposition vocabulary

The classifications are intentionally separate:

- **MVP** means the architecture MVP in Sections 4.1 and 19.1: one persistent
  local agent, the single generic loop and execution boundary, fake
  approval-controlled side effect, SQLite plus sandboxed CSV/JSON, catalog,
  sessions/events/evidence/recovery, basic memory/skills/monitor, and mock plus
  OpenAI. It proves the architecture but is not removable-v1 parity.
- **cutover** means the replacement-candidate gate definition in Section 19.2.
  Phase 9 proves PostgreSQL, a controlled real write, every retained provider,
  public-surface and legacy-package decisions, packaging, live/reliability/
  security gates, and tested fresh-start behavior. Phase 9.5 additionally
  proves the joined default-product contracts listed below. Both gates pass;
  this does not authorize Phase 10.
- **post-MVP** means deliberately deferred beyond the Phase 9 plus Phase 9.5
  replacement candidate. The surface must be documented as unavailable; it
  cannot be hidden by calling v1.

Final dispositions use only these values:

- **port** -- retain a focused leaf algorithm/translation under a named v2
  owner and protect it with a v2 regression test;
- **replace** -- retain the user-visible capability or contract through the v2
  architecture, without promising the v1 implementation or exact signature;
- **defer (documented)** -- explicitly unsupported in the candidate and named
  in migration/release documentation, with no fallback;
- **proposed removal requiring Phase 10 approval** -- not part of the target
  API and not removable until explicit human approval at Phase 10.

### Phase 9.5 replacement-readiness overlay

The stable rows below remain fully dispositioned, their Phase 9 component
evidence remains valid, and every cross-row join now passes through the
supported default composition. The candidate is eligible for human Phase 10
review, but this matrix does not authorize Phase 10.

| Closure ID | Affected stable rows | Existing owner retained | Required joined proof | Status |
| --- | --- | --- | --- | --- |
| P95-READ | MB-RR-05, MB-LG-03, MB-LG-07, MB-LG-08, MB-SP-01, MB-SP-06, API-DB-06 | Data validators, operation runtime, evidence repository, source adapters, router | Exact validator-owned source/resource/revision/sensitivity facts persist before read I/O, govern scope/routing, survive reopen, and appear in canonical acceptance/rejection metadata while adapters still revalidate | complete P9.5-02 |
| P95-MODEL | MB-SP-01 through MB-SP-05, API-ROOT-13, API-ROOT-17, API-DB-07, API-LLM-01/03 | Agent configuration, model registry/router/factory, secret provider | Persisted provider-neutral primary/fallback route reconstructs after a cold `Agent.open`/host start with secret references only and binds each operation to its route revision | complete P9.5-03 |
| P95-MONITOR | MB-MO-01 through MB-MO-07, API-DB-09 | Monitor service/scheduler, host, data-domain evidence/readiness projection | Default host enforces confirmed scope and restriction-only effective settings, evaluates every admitted condition, and records zero/one evidence-linked finding across restart/replay | complete P9.5-04 |
| P95-LEARN | MB-ML-01 through MB-ML-07, API-ROOT-11, API-DB-08 | Learning, memory, provenance, and skill services | Ordinary correction/remember and accepted-evidence inputs enter the governed learning lifecycle; later behavior changes safely; a skill proposal remains inactive until accepted | complete P9.5-05 |
| P95-EXT | MB-EX-01, API-ROOT-04, API-ROOT-08/10, SURF-05/06 | Extension and capability registries plus normal Agent/host composition | Complete — one explicit capability-provider extension composes atomically with built-ins, projects through normal context, and executes only through the sole runtime; exact durable set binding and missing/drift/collision fail-closed behavior pass in `next/tests/acceptance/test_phase_9_5_extension_composition.py`; resource-adapter/backend-provider extension categories are post-MVP | P9.5-06 complete |
| P95-UX | API-ROOT-01/03/13, API-DB-05/09, EXT-16/17, SURF-03/05/07 | Agent facade, AgentHost, private local protocol, bundled CLI | Complete — installed first-run, durable model setup, model-free serve, interactive chat, inspection, natural monitor proposal, reconnecting committed-event follow, and the clean-wheel dual-Python cold-reopen lifecycle pass through the real socket | P9.5-07 and P9.5-08 complete |

## Mandatory behavior-preservation inventory

Every bullet in Section 17.6 has one row below. Planned test names are the
minimum contract/black-box anchors; a row is not complete until its test exists
and passes at the classified gate.

### Runtime and recovery

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-RR-01 | A persisted task is claimed by at most one live lease holder. | `tests/unit/runtime/test_kernel.py::test_racing_workers_invoke_executor_at_most_once`; `tests/unit/runtime/test_store.py::test_in_memory_runtime_store_claims_task_once` | Operation runtime and task repository | `next/tests/contract/operations/test_in_memory_task_execution_lifecycle.py::test_two_concurrent_claimers_have_exactly_one_winner` | MVP | replace |
| MB-RR-02 | A stale lease holder cannot commit evidence or terminal task state. | `tests/unit/runtime/test_kernel.py::test_stale_lease_cannot_execute_or_commit` | Operation runtime, fenced lease, and evidence repositories | `next/tests/contract/operations/test_in_memory_task_execution_lifecycle.py::test_stale_fence_rejection_has_zero_checkpoint_delta` | MVP | replace |
| MB-RR-03 | Capability and executor identity are validated before execution. | `tests/unit/runtime/test_kernel.py::test_execute_task_requires_persisted_task`; `tests/unit/plugins/test_extension_registry.py::test_executor_owner_must_match_manifest_id` | Capability registry and operation runtime | `next/tests/unit/operations/test_capability_execution_boundary.py::test_execution_identity_drift_after_readiness_blocks_executor_io`; `next/tests/unit/operations/test_capability_execution_boundary.py::test_registry_rejects_executor_whose_identity_differs_from_declaration` | MVP | replace |
| MB-RR-04 | Policy denial is inspectable and invokes no executor. | `tests/unit/runtime/test_kernel.py::test_policy_block_prevents_executor_invocation`; `tests/unit/db/test_governance_runtime.py::test_execute_task_is_governance_choke_point_before_executor_runs` | Governance service and operation runtime | `next/tests/unit/operations/test_capability_execution_boundary.py::test_persisted_validation_failure_is_denied_without_executor_io`; `next/tests/acceptance/test_fake_side_effect_approval.py::test_policy_denial_is_durable_and_invokes_no_executor` | MVP | replace |
| MB-RR-05 | Evidence is accepted only for the correct task, attempt, schema, and source. | `tests/unit/runtime/test_kernel.py::test_execute_task_persists_evidence_and_correlated_events`; `tests/unit/db/test_evidence.py` | Evidence repository and operation runtime | `next/tests/unit/operations/test_capability_execution_boundary.py::test_invalid_evidence_fails_task_without_accepting_evidence`; `next/tests/unit/operations/test_runtime_blob_evidence.py::test_stale_fence_after_put_cannot_accept_blob_evidence` | MVP | replace |
| MB-RR-06 | Crash resume skips terminal tasks. | `tests/unit/db/test_agent_loop_completion_targets.py::test_resume_skips_completed_tasks_and_does_not_replay_them`; `tests/unit/runtime/test_store.py::test_sqlite_runtime_store_commits_task_success_across_restart` | Recovery service and operation runtime | `next/tests/unit/operations/test_runtime_recovery.py::test_resume_returns_existing_success_without_reexecuting`; `next/tests/acceptance/test_loop_restart_recovery.py` | MVP | replace |
| MB-RR-07 | Completion/failure events agree with persisted terminal state. | `tests/unit/runtime/test_kernel.py::test_operation_helpers_complete_and_emit_consistent_event`; `tests/unit/runtime/test_kernel.py::test_fail_operation_if_active_emits_error_and_terminal_noops` | Operation repository and committed event log | `next/tests/unit/operations/test_runtime_budgets.py::test_fail_budget_state_and_events_commit_atomically`; `next/tests/contract/storage/test_sqlite_operation_store_commit.py` | MVP | replace |
| MB-RR-08 | An unknown side-effect outcome is neither reported successful nor automatically retried. | `tests/unit/db/test_governance_runtime.py::test_expired_side_effecting_lease_requires_manual_recovery` | Operation runtime recovery classifier | `next/tests/unit/operations/test_runtime_recovery.py::test_resume_classifies_expired_unsafe_started_side_effect_as_manual`; `next/tests/unit/operations/test_capability_execution_boundary.py::test_side_effect_timeout_preserves_recovery_classification` | MVP | replace |

### Agent loop, SQL, and grounded completion

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-LG-01 | A text-only answer completes without manufacturing a task. | `daita/agents/chat/runtime.py`; `tests/unit/agents/test_chat_runtime.py` | Generic agent loop | `next/tests/acceptance/test_text_only_loop.py::test_text_only_response_completes_from_committed_runtime_state` | MVP | replace |
| MB-LG-02 | A valid tool proposal becomes a persisted task and accepted evidence before model observation. | `tests/unit/db/test_agent_loop_completion_targets.py::test_multi_turn_schema_to_sql_loop_persists_observation_before_second_turn`; `tests/unit/runtime/test_kernel.py` | Generic loop plus operation runtime | `next/tests/acceptance/test_fake_read_loop.py::test_fake_reads_follow_the_only_durable_executor_path_in_order` | MVP | replace |
| MB-LG-03 | Malformed or out-of-scope SQL fails validation before connector I/O. | `daita/db/query_sql_validation.py`; `daita/db/sql_analysis.py`; `tests/unit/db/test_plan_validation.py` | Data domain SQL validator | `next/tests/unit/domains/data/test_controller_context.py::test_invalid_sql_is_rejected_before_an_executor_request`; `next/tests/unit/domains/data/test_sql.py` | MVP | port |
| MB-LG-04 | Repair receives a structured failure reason rather than hidden deterministic replanning. | `tests/unit/db/test_agent_loop_phase2.py::test_llm_repair_falls_back_to_failed_sql_validation`; `tests/unit/db/test_agent_loop_phase2.py::test_llm_repair_blocks_without_failed_validation_evidence` | Generic loop observation path and data domain validator | `next/tests/acceptance/test_structured_action_repair.py::test_invalid_action_is_observed_then_changed_action_repairs` | MVP | replace |
| MB-LG-05 | Identical failed actions terminate through a bounded no-progress rule. | `tests/unit/db/test_agent_loop_completion_targets.py::test_repeated_failing_action_stops_with_no_progress_observation`; `tests/unit/db/test_agent_loop_phase2.py::test_llm_repair_repeated_sql_is_rejected` | Generic loop progress tracker | `next/tests/acceptance/test_structured_action_repair.py::test_repeated_normalized_failure_stops_before_more_model_or_io` | MVP | port |
| MB-LG-06 | Query scope and literal-value grounding survive follow-up turns. | `tests/unit/db/test_session_context.py::test_successful_db_run_persists_one_scope_and_follow_up_retrieves_it`; `tests/integration/from_db/test_from_db_live_production_contracts.py::test_live_literal_value_grounding_completed_vs_complete` | Session context service, catalog, and data domain | `next/tests/acceptance/test_public_agent.py::test_run_inspect_resume_and_session_transcripts_survive_reopen`; `next/tests/unit/domains/data/test_controller_context.py::test_session_history_is_ordered_and_only_projected_for_session_scope` | MVP | replace |
| MB-LG-07 | A data claim cannot complete without current applicable evidence. | `tests/unit/db/test_agent_loop_phase2.py::test_premature_finish_cannot_complete_a_data_query_without_result`; `tests/unit/db/test_agent_loop_completion_targets.py::test_finalization_policy_blocks_data_query_with_schema_evidence_only` | Data domain readiness evaluator | `next/tests/unit/domains/data/test_controller_context.py::test_valid_query_becomes_untrusted_evidence_and_response_contract` | MVP | replace |
| MB-LG-08 | Sampled, truncated, partial, or stale evidence is disclosed. | `tests/unit/db/test_result_projection.py`; `tests/unit/db/test_verification.py` | Evidence metadata and data-domain/public projection | `next/tests/unit/domains/data/test_phase4_controller.py::test_truncated_comparison_requires_explicit_partial_disclosure`; `next/tests/unit/domains/data/test_results.py` | MVP | port |
| MB-LG-09 | Cancellation, provider error, and budget exhaustion terminate honestly. | `tests/unit/runtime/test_kernel.py::test_executor_cancellation_persists_resumable_task_and_operation_state`; `tests/unit/agents/test_error_paths.py` | Generic loop and operation terminal-state owner | `next/tests/acceptance/test_loop_cancellation.py`; `next/tests/acceptance/test_loop_budgets.py`; `next/tests/acceptance/test_text_only_loop.py::test_model_failure_is_committed_once_without_whole_loop_retry` | MVP | replace |

### Approval and resume

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-AR-01 | Approval yield occurs only after task, relevant facts, and approval are persisted. | `tests/unit/runtime/test_store.py::test_sqlite_runtime_store_approval_wait_survives_restart`; `tests/unit/db/test_governance_runtime.py::test_direct_capability_execution_requires_governance_approval` | Governance service and atomic operation repository | `next/tests/acceptance/test_sqlite_update_approval_journey.py::test_public_sqlite_update_waits_reopens_and_resumes_exactly_once` | MVP | replace |
| MB-AR-02 | Approve/deny channels mutate approval state only. | `tests/unit/runtime/test_kernel.py::test_kernel_bound_approval_updates_persist_and_publish_once`; `tests/unit/db/test_governance_runtime.py::test_terminal_approval_state_cannot_later_be_approved` | Approval service | `next/tests/acceptance/test_fake_side_effect_approval.py::test_side_effect_waits_and_decision_mutates_only_approval`; `next/tests/acceptance/test_sqlite_update_approval_journey.py::test_public_sqlite_update_waits_reopens_and_resumes_exactly_once` | MVP | replace |
| MB-AR-03 | The normal operation owner resumes the existing operation. | `tests/unit/db/test_governance_runtime.py::test_resume_operation_executes_approved_blocked_task_once_and_skips_completed` | Operation runtime and host wakeup path | `next/tests/acceptance/test_fake_side_effect_approval.py::test_loop_wakes_and_resumes_the_same_operation_after_approval`; `next/tests/acceptance/test_sqlite_update_approval_journey.py::test_public_sqlite_update_waits_reopens_and_resumes_exactly_once` | MVP | replace |
| MB-AR-04 | Completed discovery/read tasks are not replayed before the approved task. | `tests/unit/db/test_governance_runtime.py::test_resume_operation_executes_approved_blocked_task_once_and_skips_completed`; `tests/unit/db/test_agent_loop_completion_targets.py::test_resume_skips_completed_tasks_and_does_not_replay_them` | Operation runtime dependency scheduler | `next/tests/acceptance/test_sqlite_update_approval_journey.py::test_public_sqlite_update_waits_reopens_and_resumes_exactly_once` | MVP | replace |
| MB-AR-05 | Denial is visible to both model and user. | `tests/unit/db/test_governance_runtime.py::test_rejected_expired_and_cancelled_approvals_remain_inspectable` | Governance observation and public projection | `next/tests/acceptance/test_fake_side_effect_approval.py::test_loop_projects_denial_to_the_model_without_executor_io`; `next/tests/unit/domains/data/test_phase7_controlled_update.py::test_denied_update_requires_no_application_disclosure_and_impact_citation` | MVP | replace |
| MB-AR-06 | Repeated resume does not duplicate the side effect. | `tests/unit/db/test_governance_runtime.py::test_concurrent_resume_claims_side_effecting_task_once`; `tests/unit/db/test_db_monitor_scheduler.py::test_approved_governed_delivery_resumes_and_sends_once` | Operation runtime idempotency and recovery owner | `next/tests/acceptance/test_fake_side_effect_approval.py::test_concurrent_resume_changes_the_marker_once`; `next/tests/acceptance/test_sqlite_update_approval_journey.py::test_public_sqlite_update_waits_reopens_and_resumes_exactly_once` | MVP | replace |
| MB-AR-07 | Approval decision racing cancellation has one deterministic outcome. | `tests/unit/runtime/test_kernel.py` approval/cancellation cases; Section 17.3 race requirement | Approval service and operation terminal transition | `next/tests/acceptance/test_fake_side_effect_approval.py::test_approval_decision_racing_cancellation_has_one_terminal_transition` | MVP | replace |

### Catalog and resource truth

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-CR-01 | Normalized resource IDs are stable for a source and native identity. | `tests/unit/catalog/test_catalog_normalizer.py`; `daita/plugins/catalog/` | Catalog identity service | `next/tests/unit/catalog/test_models.py::test_resource_identity_is_stable_and_source_scoped` | MVP | port |
| MB-CR-02 | Revisions and freshness distinguish current from stale structure. | PostgreSQL profile/freshness tests; `daita/plugins/catalog/` | Catalog revision and sync service | `next/tests/unit/catalog/test_models.py::test_structural_revision_changes_with_column_contract`; `next/tests/unit/storage/test_sqlite_catalog_refresh_semantics.py` | MVP | replace |
| MB-CR-03 | Declared and inferred relationships retain provenance and confidence. | `tests/unit/catalog/test_catalog_relationships.py`; `tests/unit/catalog/test_catalog_discoverer.py` | Catalog relationship store | `next/tests/unit/catalog/test_models.py::test_relationship_provenance_is_part_of_identity_and_connector_is_authoritative` | MVP | replace |
| MB-CR-04 | Catalog search and bounded traversal find join-relevant resources. | `tests/unit/catalog/test_catalog_relationships.py::test_catalog_find_join_path_returns_sql_ready_predicates` | Catalog search and graph traversal | `next/tests/unit/storage/test_sqlite_catalog_store.py::test_catalog_snapshot_reopens_searches_and_traverses_current_state`; `next/tests/acceptance/test_sqlite_catalog_journey.py` | MVP | replace |
| MB-CR-05 | Profile, sample, and value facts remain distinct from structural facts. | PostgreSQL catalog profile/value tests; `tests/unit/db/test_context_projection.py` | Catalog facets and projection builder | `next/tests/unit/catalog/test_models.py::test_tabular_facet_is_order_stable_and_row_estimate_is_nonstructural` | MVP | replace |
| MB-CR-06 | Sensitive classifications affect model/public projections. | `tests/unit/db/test_context_projection.py::test_public_diagnostic_and_audit_evidence_projection_modes` | Catalog classification plus projection policy | `next/tests/unit/domains/data/test_controller_context.py::test_context_projects_strictest_sensitivity_for_model_routing` | MVP | replace |
| MB-CR-07 | The data domain consumes catalog facts rather than rediscovering a private schema model. | `tests/unit/db/test_agent_loop_phase2.py::test_propose_sql_read_inserts_catalog_search_and_asset_prerequisites`; catalog ownership tests | Catalog service and data domain controller | `next/tests/unit/catalog/test_sqlite_catalog_service.py::test_service_and_data_view_preserve_scope_current_schema_and_trust`; `next/tests/unit/domains/data/test_controller_context.py` | MVP | replace |

### Memory and learning

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-ML-01 | An explicit user correction can change a later plan. | `tests/unit/db/test_memory_learning.py`; `tests/integration/from_db/test_from_db_live_memory_contracts.py::test_live_memory_metric_definition_changes_future_planning` | Learning service and memory service | `next/tests/acceptance/test_learning_journey.py::test_public_exact_correction_learning_and_revision_journey` | MVP | replace |
| MB-ML-02 | Recall filters by agent, source, resource, revision, freshness, sensitivity, and policy scope. | `tests/unit/db/test_memory_runtime.py::test_structured_db_memory_filters_before_scoring`; live source-scope/stale test | Memory query service | `next/tests/unit/memory/test_service.py::test_recall_filters_authority_facts_before_deterministic_ranking` | MVP | port |
| MB-ML-03 | Stale or superseded memory is excluded or clearly qualified. | `tests/unit/db/test_memory_runtime.py::test_contract_projection_downgrades_cross_source_stale_and_low_confidence` | Memory version/freshness service and context projection | `next/tests/unit/memory/test_service.py::test_list_and_inspect_keep_stale_expired_and_history_visible`; `next/tests/contract/storage/test_sqlite_phase5_state.py::test_memory_filters_history_cas_restore_and_corruption` | MVP | replace |
| MB-ML-04 | Memory provenance resolves to an originating user statement or accepted evidence. | `tests/unit/db/test_memory_learning.py::test_learner_promotes_safe_unit_candidate_through_memory_write`; context-projection provenance tests | Learning proposal validator and memory store | `next/tests/acceptance/test_learning_journey.py::test_public_exact_correction_learning_and_revision_journey` | MVP | replace |
| MB-ML-05 | A policy-blocked or failed action is not learned as an enforceable success. | `tests/unit/db/test_memory_learning.py::test_learning_enqueue_gates_skip_ineligible_operations`; `tests/unit/db/test_memory_runtime.py::test_contract_projection_blocks_policy_denied_refs` | Learning eligibility policy | `next/tests/unit/memory/test_learning_service.py::test_learning_consumes_only_completed_succeeded_operations` | MVP | replace |
| MB-ML-06 | Raw sensitive rows are not silently promoted into durable memory. | `tests/unit/db/test_memory_learning.py::test_learner_rejects_duplicate_missing_source_cross_source_and_pii_candidates`; live PII rejection test | Learning safety validator | `next/tests/unit/memory/test_learning_service.py::test_unsafe_candidates_persist_only_a_redacted_rejection`; `next/tests/acceptance/test_learning_journey.py::test_public_exact_correction_learning_and_revision_journey` | MVP | replace |
| MB-ML-07 | Memory remains inspectable, correctable, and reversible after restart. | `tests/unit/db/test_memory_runtime.py::test_structured_db_memory_survives_local_backend_reconstruction`; memory lifecycle tests | Memory store and public inspection API | `next/tests/acceptance/test_learning_journey.py::test_public_exact_correction_learning_and_revision_journey` | MVP | replace |

### Monitors

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-MO-01 | Create, list, inspect, pause, resume, run-now, and delete lifecycle is durable. | `tests/unit/db/test_db_monitors.py::test_db_agent_typed_monitor_crud_records_runtime_operations`; `tests/unit/db/test_db_monitor_commands.py` | Monitor service and monitor repository | `next/tests/acceptance/test_monitor_lifecycle.py::test_public_monitor_lifecycle_survives_restart` | MVP | replace |
| MB-MO-02 | Two schedulers cannot claim the same due tick. | `tests/unit/db/test_db_monitor_scheduler.py::test_two_schedulers_share_one_lease_and_only_one_triggers`; live tick-lease test | Monitor scheduler and fenced tick repository | `next/tests/contract/monitors/test_sqlite_monitor_store.py::test_scheduled_claim_race_and_fenced_outcome_are_atomic` | MVP | replace |
| MB-MO-03 | Checkpoint/cursor progress commits with the monitor outcome. | `tests/unit/db/test_db_monitor_scheduler.py::test_run_commit_rejects_stale_monitor_or_state_snapshot` | Monitor repository atomic outcome commit | `next/tests/unit/monitors/test_scheduler.py::test_due_match_uses_one_stable_normal_trigger_and_atomic_outcome`; `next/tests/unit/monitors/test_store_contract.py::test_outcome_commit_guards_fence_run_links_events_and_evidence_finding` | MVP | replace |
| MB-MO-04 | Cooldown, backoff, and missed-run policy survive restart. | `tests/unit/db/test_db_monitor_scheduler.py::test_scheduler_respects_pause_cooldown_and_backoff_gates`; live cooldown durability test | Monitor scheduler | `next/tests/unit/monitors/test_scheduler.py::test_failure_applies_backoff_and_catches_up_once_after_downtime`; `next/tests/contract/monitors/test_sqlite_monitor_store.py::test_lifecycle_events_and_run_now_survive_reopen` | MVP | replace |
| MB-MO-05 | One due occurrence creates at most one ordinary operation. | `tests/unit/db/test_db_monitor_scheduler.py::test_triggered_tick_creates_generic_operation_and_counts_consecutive_matches`; repeated live scheduler tick test | Trigger inbox, monitor scheduler, and operation service | `next/tests/unit/monitors/test_scheduler.py::test_due_match_uses_one_stable_normal_trigger_and_atomic_outcome`; `next/tests/unit/hosting/test_host.py::test_monitor_run_once_uses_the_ordinary_exact_trigger_path` | MVP | replace |
| MB-MO-06 | Monitor-triggered writes use ordinary policy and approval. | `tests/unit/db/test_db_monitor_scheduler.py::test_monitor_write_governance_uses_task_from_plan_task_specs`; live governed-write monitor test | Monitor service plus ordinary operation runtime | `next/tests/unit/monitors/test_scheduler.py::test_waiting_run_now_reclaims_same_trigger_after_approval_wake`; `next/tests/architecture/test_phase6_monitor_architecture.py::test_monitor_package_has_no_alternate_execution_or_provider_path` | MVP | replace |
| MB-MO-07 | A monitor never bypasses loop readiness, task evidence, or operation audit. | `tests/unit/db/test_db_monitor_scheduler.py::test_monitor_scheduler_does_not_bypass_runtime_execution_boundaries`; live observation contract | Generic loop and operation runtime; monitor only emits triggers | `next/tests/unit/monitors/test_scheduler.py::test_unaccepted_projection_evidence_cannot_commit_monitor_outcome`; `next/tests/architecture/test_phase6_monitor_architecture.py::test_monitor_package_has_no_alternate_execution_or_provider_path` | MVP | replace |

### Sessions, providers, and projections

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-SP-01 | Stateful follow-up uses only the intended session context. | `tests/unit/db/test_session_context.py`; `tests/integration/from_db/test_from_db_live_production_contracts.py::test_live_stateful_followup_uses_session_context` | Session service and context builder | `next/tests/acceptance/test_public_agent.py::test_run_inspect_resume_and_session_transcripts_survive_reopen`; `next/tests/unit/domains/data/test_controller_context.py::test_session_history_is_ordered_and_only_projected_for_session_scope` | MVP | replace |
| MB-SP-02 | Concurrent sessions cannot leak messages, scope, approvals, or evidence. | `tests/unit/db/test_session_context.py`; live stateless non-leak test | Session repository, context builder, and projection policy | `next/tests/acceptance/test_public_agent.py::test_run_inspect_resume_and_session_transcripts_survive_reopen`; `next/tests/unit/domains/data/test_controller_context.py::test_session_history_is_ordered_and_only_projected_for_session_scope` | MVP | replace |
| MB-SP-03 | Every advertised provider normalizes messages, tool calls, streaming deltas, stop reasons, usage, and retryable errors to canonical types. | `tests/unit/llm/`; `tests/integration/llm/test_llm_providers_live.py` | Model provider adapters and provider conformance suite | `next/tests/contract/models/test_provider_conformance.py::test_advertised_provider_text_usage_and_structured_output`; `next/tests/contract/models/test_provider_conformance.py::test_advertised_provider_streams_normalize_tool_arguments`; `next/tests/contract/models/test_provider_conformance.py::test_advertised_provider_normalizes_errors_and_cancellation`; `next/tests/live/test_model_providers_live.py::test_retained_provider_live_text_conformance` | cutover | port |
| MB-SP-04 | Provider switching continues from canonical state; provider-native transcripts are not authoritative. | provider live tests; v1 canonical message translation in `daita/llm/` | Model router and canonical session/operation stores | `next/tests/acceptance/test_provider_continuity.py::test_provider_switch_uses_canonical_context_and_survives_reopen` | cutover | replace |
| MB-SP-05 | Missing optional SDKs preserve minimal import and report the documented install hint when selected. | `tests/unit/llm/test_provider_lifecycle.py`; plugin lazy-import tests | Packaging plus lazy provider/adapter factories | `next/tests/contract/models/test_provider_conformance.py::test_advertised_provider_lazy_import_uses_exact_extra_hint`; `next/tests/contract/models/test_registry.py::test_factory_constructs_retained_adapters_without_sdk_io`; `next/tests/architecture/test_phase8_model_architecture.py::test_optional_provider_sdks_are_not_imported_at_module_scope` | cutover | port |
| MB-SP-06 | Audit, model, and public projections differ according to sensitivity and retention rules. | `tests/unit/db/test_context_projection.py`; `tests/unit/db/test_result_projection.py` | Projection policy and context builder | `next/tests/unit/domains/data/test_controller_context.py::test_context_projects_strictest_sensitivity_for_model_routing`; `next/tests/unit/hosting/test_local_server.py::test_local_governance_projection_exposes_safe_facts_and_approval_hashes` | MVP | port |

## Root public-export dispositions

Every name in root `daita.__all__`, `daita.db.__all__`, and
`daita.llm.__all__` at the v1 baseline appears in this section. Grouping does
not imply one v2 class per v1 name; v2 keeps the behavior only where the target
architecture assigns a coherent owner.

| Stable ID | v1 public names / feature | Intended v2 owner and target surface | Planned v2 test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- |
| API-ROOT-01 | `Agent` | Thin persistent `Agent.create/open/attach/run/stream/inspect/approve/reject/cancel/resume` facade | `next/tests/acceptance/test_public_agent.py` | MVP | replace |
| API-ROOT-02 | `BaseAgent` | Composition through `Agent`, domain-controller, and service protocols; no subclass framework | `next/tests/architecture/test_phase2_agent_architecture.py::test_public_agent_is_a_thin_embedded_facade` | cutover | proposed removal requiring Phase 10 approval |
| API-ROOT-03 | `ConversationHistory` | Session/message service and canonical operation transcript | `next/tests/acceptance/test_public_agent.py::test_run_inspect_resume_and_session_transcripts_survive_reopen` | MVP | replace |
| API-ROOT-04 | `tool` | Explicit local capability declaration projected as a `ToolView`; never direct loop execution | `next/tests/contract/extensions/test_local_capability.py` | cutover | replace |
| API-ROOT-05 | `configure_tracing`, `get_trace_manager`, `set_trace_context` | Event-to-telemetry projection and optional exporter registration | `next/tests/contract/telemetry/test_event_projection.py` | cutover | replace |
| API-ROOT-06 | `postgresql`, `sqlite` | Built-in resource adapters and declared executors | `next/tests/contract/domains/data/test_relational_read_conformance.py`; `next/tests/acceptance/test_postgresql_catalog_journey.py`; `next/tests/live/test_postgresql_live.py::test_live_postgresql_catalog_and_bounded_read_are_least_privileged` | cutover | replace |
| API-ROOT-07 | `mysql`, `mongodb`, `rest`, `s3`, `slack`, `elasticsearch`, `redis_messaging` | Future explicit resource/capability adapters; absent from candidate runtime | Phase 9 absence/install-hint and support-matrix checks | post-MVP | defer (documented) |
| API-ROOT-08 | `BasePlugin`, `ConnectorPlugin`, `DomainServicePlugin`, `ObservabilityPlugin`, `RuntimeExtensionPlugin`, `SkillPlugin`, `WorkerProviderPlugin` | Narrow capability-provider manifest protocol in the MVP; resource adapters, backend providers, and observers remain explicit post-MVP categories | `next/tests/contract/extensions/test_extension_registry.py`; `next/tests/acceptance/test_phase_9_5_extension_composition.py`; `next/tests/architecture/test_phase1_loop_architecture.py::test_generic_loop_imports_contracts_not_domain_or_provider_implementations` | cutover | proposed removal requiring Phase 10 approval |
| API-ROOT-09 | `EmptySecretProvider`, `SecretProvider` | Injectable secret-provider protocol plus explicit empty/env/keychain implementations | `next/tests/contract/security/test_secret_provider.py` | cutover | replace |
| API-ROOT-10 | `PluginContext`, `PluginKind`, `PluginManifest`, `ServiceRegistry`, `ExtensionRegistry`, `RegistryDiagnostic` | Validated narrow extension manifest/registry and diagnostics | `next/tests/contract/extensions/test_extension_registry.py` | cutover | replace |
| API-ROOT-11 | `BaseSkill`, `Skill`, `SkillActivation`, `SkillActivationRules`, `SkillDiscovery`, `SkillResolver`, `SkillResolution` | Versioned `SKILL.md` index, selection, activation, and service records | `next/tests/unit/skills/test_skill_service.py`; `next/tests/acceptance/test_skill_change_lifecycle.py::test_public_skill_change_propose_accept_reopen_lifecycle` | MVP | replace |
| API-ROOT-12 | `SkillRuntimeEffects` | Skills may reference capabilities but cannot declare runtime effects/executors | `next/tests/unit/skills/test_skill_boundaries.py`; `next/tests/architecture/test_phase5_context_learning_architecture.py` | cutover | proposed removal requiring Phase 10 approval |
| API-ROOT-13 | `AgentConfig`, `RetryPolicy`, `RetryStrategy` | Versioned agent/model/budget/policy configuration owned by services and router | `next/tests/contract/config/test_agent_configuration.py` | cutover | replace |
| API-ROOT-14 | `apply_focus` | Bounded capability arguments and evidence projections; the standalone v1 Focus DSL is not an MVP owner | Phase 9 migration/support-matrix assertion | post-MVP | proposed removal requiring Phase 10 approval |
| API-ROOT-15 | `DaitaError`, `AgentError`, `LLMError`, `ConfigError`, `PluginError`, `SkillError`, `TransientError`, `RetryableError`, `PermanentError`, `RateLimitError`, `AuthenticationError`, `ValidationError`, `FocusDSLError`, `DataQualityError` | Stable typed public result/error taxonomy, normalized provider errors, and subsystem-specific details | `next/tests/contract/test_errors.py` | cutover | replace |
| API-ROOT-16 | `ItemAssertion` | Future data-quality capability provider; not part of architecture MVP | Phase 9 documented-absence assertion | post-MVP | defer (documented) |
| API-ROOT-17 | `create_llm_provider` | Model registry/router configured by canonical provider/model profiles | `next/tests/contract/models/test_registry.py` | MVP | replace |
| API-ROOT-18 | `BaseEmbeddingProvider` | Separate embedding-provider protocol; lexical FTS is authoritative for MVP | Phase 9 documented-absence assertion | post-MVP | defer (documented) |
| API-ROOT-19 | `__version__` | Package version metadata | `next/tests/architecture/test_import_firewall.py::test_pytest_import_resolves_only_to_v2_source`; `next/tests/architecture/test_phase0_constitution.py::test_isolated_distribution_metadata_matches_the_constitution` | cutover | port |
| API-DB-01 | `from_db`, `DbAgent` | `Agent.create/open`, `attach(SQLiteSource/PostgreSQLSource)`, then the same `Agent.run`; no DB agent subclass | `next/tests/acceptance/test_sqlite_catalog_journey.py`; `next/tests/acceptance/test_postgresql_catalog_journey.py`; `next/tests/live/test_postgresql_live.py::test_live_postgresql_catalog_and_bounded_read_are_least_privileged` | cutover | replace |
| API-DB-02 | `DbRuntime` | Generic operation runtime plus built-in data-domain controller | `next/tests/architecture/test_phase1_loop_architecture.py::test_operation_runtime_is_the_only_executor_invocation_boundary` | MVP | proposed removal requiring Phase 10 approval |
| API-DB-03 | `DbIntent`, `DbIntentKind`, `DbRequest`, `DbOperationContract`, `DbOperationResult` | Canonical trigger/operation/action/readiness/exit and public result records | `next/tests/unit/loop/test_loop_models.py` | MVP | replace |
| API-DB-04 | `DbLimits`, `DbRuntimeConfig`, `DbRuntimeOptions`, `DbExecutionConfig` | Agent, loop-budget, operation-runtime, store, and host configuration | `next/tests/contract/config/test_runtime_configuration.py` | cutover | replace |
| API-DB-05 | `DbRuntimeInspection` | Generic operation/agent inspection views | `next/tests/acceptance/test_public_agent.py::test_run_inspect_resume_and_session_transcripts_survive_reopen` | MVP | replace |
| API-DB-06 | `DbSourceOptions` | Typed SQLite/PostgreSQL source configuration and scope policy | `next/tests/unit/adapters/test_sqlite.py`; `next/tests/unit/adapters/test_postgresql.py`; `next/tests/contract/domains/data/test_relational_read_conformance.py` | cutover | replace |
| API-DB-07 | `DbLLMConfig` | Canonical model profiles and router policy | `next/tests/unit/llm/test_model_profiles.py`; `next/tests/unit/llm/test_routing.py` | cutover | replace |
| API-DB-08 | `DbMemoryConfig` | Memory service policy/configuration | `next/tests/unit/memory/test_models.py`; `next/tests/unit/memory/test_service.py` | MVP | replace |
| API-DB-09 | `DbMonitor`, `DbMonitorInspection`, `DbMonitorMutation`, `DbMonitorRun`, `DbMonitorState`, `DbMonitorStore` | Generic monitor service, records, repository, scheduler, and inspection API | `next/tests/acceptance/test_monitor_lifecycle.py` | MVP | replace |
| API-DB-10 | `HostedInAppMonitorDeliveryPlugin` | Local durable finding/event; outbound delivery is a later explicit extension | Phase 9 documented-absence assertion | post-MVP | defer (documented) |
| API-LLM-01 | `register_llm_provider`, `list_available_providers`, `BaseLLMProvider` | Canonical model-provider protocol and explicit registry | `next/tests/contract/models/test_registry.py` | cutover | replace |
| API-LLM-02 | `CostEstimate`, `ModelPricing`, `TokenUsage`, `estimate_llm_cost` | Canonical usage/cost records and router budget estimator | `next/tests/unit/llm/test_llm_models.py::test_usage_accounts_exact_decimal_cost_without_float_drift`; `next/tests/unit/llm/test_routing.py::test_primary_route_records_attempt_and_profile_owned_decimal_cost` | cutover | port |
| API-LLM-03 | `OpenAIProvider`, `AnthropicProvider`, `GrokProvider`, `GeminiProvider`, `MockLLMProvider`; factory-only `OllamaProvider`; supported OpenAI-compatible endpoints | Lazy model adapters satisfying the shared provider conformance suite | `next/tests/contract/models/test_provider_conformance.py`; `next/tests/live/test_model_providers_live.py::test_retained_provider_live_text_conformance` | cutover | port |

The 61 root names, 23 database names, and 13 LLM names are all named above.
`create_llm_provider` appears in both root and LLM exports and is intentionally
accounted for once by `API-ROOT-17`; the LLM row covers the remaining registry
surface.

## Optional-extra and advertised integration dispositions

All 44 root optional-extra names at the baseline are explicitly listed below.
Bundle extras inherit the dispositions of their members; they never make a
deferred integration available through v1 fallback.

| Stable ID | v1 extra(s) / advertised area | Candidate posture | Classification | Final disposition |
| --- | --- | --- | --- | --- |
| EXT-01 | `dev` | V2-only test/build/type/format dependencies; isolated metadata is asserted in `next/tests/architecture/test_phase0_constitution.py` | MVP | replace |
| EXT-02 | `sqlite` | Required architecture-MVP adapter and data-domain conformance in `next/tests/contract/domains/data/test_relational_read_conformance.py` and `next/tests/acceptance/test_sqlite_catalog_journey.py` | MVP | replace |
| EXT-03 | `postgresql` | Replacement-candidate adapter through the same contracts; deterministic conformance passes in `next/tests/contract/domains/data/test_relational_read_conformance.py` and `next/tests/acceptance/test_postgresql_catalog_journey.py`, and real least-privileged service acceptance passes in `next/tests/live/test_postgresql_live.py::test_live_postgresql_catalog_and_bounded_read_are_least_privileged` | cutover | replace |
| EXT-04 | `anthropic`, `google`, `llm-all` | Retained model adapters (`google` maps to Gemini) pass shared deterministic conformance in `next/tests/contract/models/test_provider_conformance.py` and real boundary acceptance in `next/tests/live/test_model_providers_live.py::test_retained_provider_live_text_conformance`; bundle-extra metadata is asserted separately | cutover | replace |
| EXT-05 | `transformers` | V1 local Transformers model path is not the retained local-provider contract; Ollama/OpenAI-compatible is | post-MVP | defer (documented) |
| EXT-06 | `data` | Phase 4 completes the v2-native bounded, sandboxed CSV/JSON path without pandas in `next/tests/acceptance/test_local_file_comparison_journey.py`; remaining cutover work is data-extra/API parity, with XLSX optional only if inexpensive | cutover | replace |
| EXT-07 | `memory` | Memory is a core lifecycle using SQLite FTS as proven by `next/tests/unit/memory/test_service.py` and `next/tests/acceptance/test_learning_journey.py`; embedding/graph packages are not required semantics | MVP | replace |
| EXT-08 | `sentence-transformers`, `voyage` | Optional embedding accelerators | post-MVP | defer (documented) |
| EXT-09 | `data-quality`, `lineage` | Future capability providers, not alternate runtimes | post-MVP | defer (documented) |
| EXT-10 | `mysql`, `mongodb`, `snowflake`, `bigquery`, `elasticsearch`, `opensearch`, `databases` | Additional databases/search sources after SQLite/PostgreSQL contracts are stable | post-MVP | defer (documented) |
| EXT-11 | `chromadb`, `pinecone`, `qdrant`, `vectordb`, `neo4j` | Vector/graph acceleration is not required for catalog or memory semantics | post-MVP | defer (documented) |
| EXT-12 | `redis` | Distributed state/messaging is not part of the local one-writer MVP | post-MVP | defer (documented) |
| EXT-13 | `aws`, `azure`, `gcp`, `google-drive`, `cloud` | Cloud/object/document resource adapters are on the post-MVP roadmap | post-MVP | defer (documented) |
| EXT-14 | `github`, `slack`, `mcp` | App/communication/protocol integrations are explicit later adapters/delivery extensions | post-MVP | defer (documented) |
| EXT-15 | `web`, `websearch`, `exa` | Web content/search adapters are outside the data-source MVP | post-MVP | defer (documented) |
| EXT-16 | `cli` | The candidate package is the sole Daita 2.0 console owner. Its bundled `daita` CLI delegates local work to `AgentHost` through the public host/client boundary, as proven by `next/tests/unit/test_cli.py`, `next/tests/unit/hosting/test_local_server.py`, and `next/tests/contract/packaging/test_candidate_metadata.py`; the separate v1 CLI package is not a dependency | cutover | replace |
| EXT-17 | `api-server`, `production` | Local host API/serve behavior is ported in `next/tests/unit/hosting/test_host.py` and `next/tests/unit/hosting/test_local_server.py`; generic v1 deployment bundles are not candidate runtime dependencies | cutover | replace |
| EXT-18 | `otlp` | `next/tests/contract/telemetry/test_event_projection.py` replaces global tracing state with a failure-isolated exporter boundary over redacted committed events; a concrete OTLP transport adapter is deferred | post-MVP | defer (documented) |
| EXT-19 | `recommended`, `complete`, `all` | Rebuilt only from supported v2 extras, with exact deduplicated bundle contracts and deferred-family exclusion in `next/tests/contract/packaging/test_candidate_metadata.py::test_bundle_extras_are_deduplicated_unions_of_supported_atomic_extras` and `next/tests/contract/packaging/test_candidate_metadata.py::test_deferred_v1_integrations_are_not_advertised_by_any_v2_extra` | cutover | replace |

Counted names: `dev`; `sqlite`; `postgresql`; `anthropic`; `google`;
`llm-all`; `transformers`; `data`; `memory`; `sentence-transformers`;
`voyage`; `data-quality`; `lineage`; `mysql`; `mongodb`; `snowflake`;
`bigquery`; `elasticsearch`; `opensearch`; `databases`; `chromadb`;
`pinecone`; `qdrant`; `vectordb`; `neo4j`; `redis`; `aws`; `azure`; `gcp`;
`google-drive`; `cloud`; `github`; `slack`; `mcp`; `web`; `websearch`;
`exa`; `cli`; `api-server`; `production`; `otlp`; `recommended`; `complete`;
`all`.

## Advertised examples and secondary surfaces

| Stable ID | v1 advertised surface | Candidate posture | Classification | Final disposition |
| --- | --- | --- | --- | --- |
| SURF-01 | Examples `00_quickstart_sqlite_from_db.py`, `01_inspectable_operation.py`, `02_catalog_assisted_joins.py`, `03_governed_reads_and_writes.py`, `04_persistent_runtime_store.py`, `06_memory_for_business_semantics.py`, `07_monitor_orders.py`, `09_custom_data_plugin_extension.py`, `10_csv_to_sqlite_data_app.py` | V2-native retained examples run through the public API/host contracts and new evidence model; `next/tests/contract/packaging/test_examples.py::test_every_retained_example_has_an_executable_help_path`, `next/tests/contract/packaging/test_examples.py::test_quickstart_runs_safely_offline_with_an_explicit_fresh_root`, and `next/tests/contract/packaging/test_examples.py::test_examples_use_v2_surfaces_without_legacy_imports_or_auto_approval` | cutover | replace |
| SURF-02 | Examples `05_data_quality_and_lineage.py`, `08_infrastructure_catalog.py` | Document as deferred with their capability/resource adapters | post-MVP | defer (documented) |
| SURF-03 | `examples/deployments/data-team-agent/` | Production-shaped local-host deployment uses the private Unix-socket protocol and has executable dry-run coverage in `next/tests/contract/packaging/test_examples.py::test_local_host_deployment_has_safe_help_and_dry_run` | cutover | replace |
| SURF-04 | `daita.evals` developer-preview API and structured eval artifacts | Keep neutral datasets/fixtures where useful; v1 eval framework is not a candidate runtime dependency | post-MVP | defer (documented) |
| SURF-05 | Generic local tool agents, streaming, and detailed diagnostics | Single loop, explicit capabilities, canonical committed events, and public inspection results exercised through `next/tests/acceptance/test_public_agent.py` and `next/tests/contract/extensions/test_local_capability.py` | cutover | replace |
| SURF-06 | Direct plugin context managers such as `async with sqlite(...)` | Explicit adapter lifecycle remains available only if it does not bypass the operation runtime for agent-owned work; `next/tests/architecture/test_phase4_non_database_architecture.py` and `next/tests/unit/adapters/test_sqlite.py` enforce the boundary | cutover | replace |
| SURF-07 | External `daita-cli` and `daita-client` packages | Explicit compatibility decision: retire both legacy packages from the Daita 2.0 support surface without modifying their repositories. Their local responsibilities are replaced by the candidate-owned `daita` CLI, `Agent`, `AgentHost`, and local client contracts proven in `next/tests/unit/test_cli.py`, `next/tests/unit/hosting/test_local_protocol.py`, and `next/tests/unit/hosting/test_local_server.py`; no legacy fallback or co-install guarantee is advertised | post-MVP | defer (documented) |

## Completion accounting

- Mandatory behavior rows: **51 of 51 classified**.
- Root `daita.__all__`: **61 of 61 names dispositioned**.
- Root `daita.db.__all__`: **23 of 23 names dispositioned**.
- Root `daita.llm.__all__`: **13 of 13 names dispositioned** (including the
  duplicate root LLM factory by cross-reference).
- Root optional extras: **44 of 44 names dispositioned**.
- Stable matrix rows: **77 of 77 finalized** — 56 replace, 8 port, and 13
  defer (documented); no external-integration disposition remains.
- Test-file-level classification is maintained separately in
  `TEST_DISPOSITION.csv` and validated against Git-tracked root tests.
- The Phase 9 gate resolves every individual port/replace row to an executable
  passing v2 anchor. The Phase 9.5 overlay now records passing joined default-
  composition evidence, so those rows collectively establish replacement
  readiness. Deferred rows remain named in support and breaking-change
  documentation and have no v1 fallback. Phase 10 remains separately gated.
