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
- **cutover** means the replacement-candidate gate in Section 19.2, completed
  by Phase 9: PostgreSQL, a controlled real write, every retained provider,
  public-surface dispositions, external CLI/client integration, packaging,
  live/reliability/security gates, and tested fresh-start or migration
  behavior. It still does not authorize Phase 10.
- **post-MVP** means deliberately deferred beyond the Phase 9 replacement
  candidate. The surface must be documented as unavailable; it cannot be
  hidden by calling v1.

Final dispositions use only these values:

- **port** -- retain a focused leaf algorithm/translation under a named v2
  owner and protect it with a v2 regression test;
- **replace** -- retain the user-visible capability or contract through the v2
  architecture, without promising the v1 implementation or exact signature;
- **defer (documented)** -- explicitly unsupported in the candidate and named
  in migration/release documentation, with no fallback;
- **external integration** -- owned by a separately packaged client, CLI, or
  adapter integrating only through proven v2 contracts; and
- **proposed removal requiring Phase 10 approval** -- not part of the target
  API and not removable until explicit human approval at Phase 10.

## Mandatory behavior-preservation inventory

Every bullet in Section 17.6 has one row below. Planned test names are the
minimum contract/black-box anchors; a row is not complete until its test exists
and passes at the classified gate.

### Runtime and recovery

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-RR-01 | A persisted task is claimed by at most one live lease holder. | `tests/unit/runtime/test_kernel.py::test_racing_workers_invoke_executor_at_most_once`; `tests/unit/runtime/test_store.py::test_in_memory_runtime_store_claims_task_once` | Operation runtime and task repository | `next/tests/contract/runtime/test_task_leases.py::test_two_workers_claim_one_task_once` | MVP | replace |
| MB-RR-02 | A stale lease holder cannot commit evidence or terminal task state. | `tests/unit/runtime/test_kernel.py::test_stale_lease_cannot_execute_or_commit` | Operation runtime, fenced lease, and evidence repositories | `next/tests/contract/runtime/test_task_leases.py::test_stale_fence_cannot_commit` | MVP | replace |
| MB-RR-03 | Capability and executor identity are validated before execution. | `tests/unit/runtime/test_kernel.py::test_execute_task_requires_persisted_task`; `tests/unit/plugins/test_extension_registry.py::test_executor_owner_must_match_manifest_id` | Capability registry and operation runtime | `next/tests/contract/runtime/test_execution_boundary.py::test_identity_mismatch_prevents_executor_io` | MVP | replace |
| MB-RR-04 | Policy denial is inspectable and invokes no executor. | `tests/unit/runtime/test_kernel.py::test_policy_block_prevents_executor_invocation`; `tests/unit/db/test_governance_runtime.py::test_execute_task_is_governance_choke_point_before_executor_runs` | Governance service and operation runtime | `next/tests/contract/runtime/test_governance.py::test_denial_persists_result_without_executor_call` | MVP | replace |
| MB-RR-05 | Evidence is accepted only for the correct task, attempt, schema, and source. | `tests/unit/runtime/test_kernel.py::test_execute_task_persists_evidence_and_correlated_events`; `tests/unit/db/test_evidence.py` | Evidence repository and operation runtime | `next/tests/contract/runtime/test_evidence_acceptance.py::test_mismatched_evidence_is_rejected` | MVP | replace |
| MB-RR-06 | Crash resume skips terminal tasks. | `tests/unit/db/test_agent_loop_completion_targets.py::test_resume_skips_completed_tasks_and_does_not_replay_them`; `tests/unit/runtime/test_store.py::test_sqlite_runtime_store_commits_task_success_across_restart` | Recovery service and operation runtime | `next/tests/acceptance/test_restart_recovery.py::test_resume_skips_terminal_tasks` | MVP | replace |
| MB-RR-07 | Completion/failure events agree with persisted terminal state. | `tests/unit/runtime/test_kernel.py::test_operation_helpers_complete_and_emit_consistent_event`; `tests/unit/runtime/test_kernel.py::test_fail_operation_if_active_emits_error_and_terminal_noops` | Operation repository and committed event log | `next/tests/contract/state/test_terminal_event_atomicity.py::test_terminal_state_and_event_agree` | MVP | replace |
| MB-RR-08 | An unknown side-effect outcome is neither reported successful nor automatically retried. | `tests/unit/db/test_governance_runtime.py::test_expired_side_effecting_lease_requires_manual_recovery` | Operation runtime recovery classifier | `next/tests/contract/runtime/test_side_effect_recovery.py::test_unknown_outcome_requires_manual_recovery` | MVP | replace |

### Agent loop, SQL, and grounded completion

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-LG-01 | A text-only answer completes without manufacturing a task. | `daita/agents/chat/runtime.py`; `tests/unit/agents/test_chat_runtime.py` | Generic agent loop | `next/tests/contract/loop/test_scripted_trajectories.py::test_text_only_completion_has_no_tasks` | MVP | replace |
| MB-LG-02 | A valid tool proposal becomes a persisted task and accepted evidence before model observation. | `tests/unit/db/test_agent_loop_completion_targets.py::test_multi_turn_schema_to_sql_loop_persists_observation_before_second_turn`; `tests/unit/runtime/test_kernel.py` | Generic loop plus operation runtime | `next/tests/contract/loop/test_scripted_trajectories.py::test_action_task_evidence_observation_order` | MVP | replace |
| MB-LG-03 | Malformed or out-of-scope SQL fails validation before connector I/O. | `daita/db/query_sql_validation.py`; `daita/db/sql_analysis.py`; `tests/unit/db/test_plan_validation.py` | Data domain SQL validator | `next/tests/contract/data/test_sql_guardrails.py::test_invalid_or_out_of_scope_sql_never_reaches_adapter` | MVP | port |
| MB-LG-04 | Repair receives a structured failure reason rather than hidden deterministic replanning. | `tests/unit/db/test_agent_loop_phase2.py::test_llm_repair_falls_back_to_failed_sql_validation`; `tests/unit/db/test_agent_loop_phase2.py::test_llm_repair_blocks_without_failed_validation_evidence` | Generic loop observation path and data domain validator | `next/tests/acceptance/test_grounded_query.py::test_validation_failure_is_observed_then_repaired_by_model` | MVP | replace |
| MB-LG-05 | Identical failed actions terminate through a bounded no-progress rule. | `tests/unit/db/test_agent_loop_completion_targets.py::test_repeated_failing_action_stops_with_no_progress_observation`; `tests/unit/db/test_agent_loop_phase2.py::test_llm_repair_repeated_sql_is_rejected` | Generic loop progress tracker | `next/tests/contract/loop/test_no_progress.py::test_identical_failure_stops_early` | MVP | port |
| MB-LG-06 | Query scope and literal-value grounding survive follow-up turns. | `tests/unit/db/test_session_context.py::test_successful_db_run_persists_one_scope_and_follow_up_retrieves_it`; `tests/integration/from_db/test_from_db_live_production_contracts.py::test_live_literal_value_grounding_completed_vs_complete` | Session context service, catalog, and data domain | `next/tests/acceptance/test_session_grounding.py::test_follow_up_retains_scope_and_literal_grounding` | MVP | replace |
| MB-LG-07 | A data claim cannot complete without current applicable evidence. | `tests/unit/db/test_agent_loop_phase2.py::test_premature_finish_cannot_complete_a_data_query_without_result`; `tests/unit/db/test_agent_loop_completion_targets.py::test_finalization_policy_blocks_data_query_with_schema_evidence_only` | Data domain readiness evaluator | `next/tests/contract/data/test_readiness.py::test_data_claim_requires_current_applicable_evidence` | MVP | replace |
| MB-LG-08 | Sampled, truncated, partial, or stale evidence is disclosed. | `tests/unit/db/test_result_projection.py`; `tests/unit/db/test_verification.py` | Evidence metadata and data-domain/public projection | `next/tests/acceptance/test_grounded_query.py::test_bounded_evidence_limitations_are_disclosed` | MVP | port |
| MB-LG-09 | Cancellation, provider error, and budget exhaustion terminate honestly. | `tests/unit/runtime/test_kernel.py::test_executor_cancellation_persists_resumable_task_and_operation_state`; `tests/unit/agents/test_error_paths.py` | Generic loop and operation terminal-state owner | `next/tests/contract/loop/test_termination.py::test_cancel_provider_error_and_budget_have_typed_honest_exits` | MVP | replace |

### Approval and resume

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-AR-01 | Approval yield occurs only after task, relevant facts, and approval are persisted. | `tests/unit/runtime/test_store.py::test_sqlite_runtime_store_approval_wait_survives_restart`; `tests/unit/db/test_governance_runtime.py::test_direct_capability_execution_requires_governance_approval` | Governance service and atomic operation repository | `next/tests/contract/runtime/test_approval_resume.py::test_wait_checkpoint_contains_task_facts_and_approval` | MVP | replace |
| MB-AR-02 | Approve/deny channels mutate approval state only. | `tests/unit/runtime/test_kernel.py::test_kernel_bound_approval_updates_persist_and_publish_once`; `tests/unit/db/test_governance_runtime.py::test_terminal_approval_state_cannot_later_be_approved` | Approval service | `next/tests/contract/runtime/test_approval_resume.py::test_decision_channel_does_not_execute_or_replan` | MVP | replace |
| MB-AR-03 | The normal operation owner resumes the existing operation. | `tests/unit/db/test_governance_runtime.py::test_resume_operation_executes_approved_blocked_task_once_and_skips_completed` | Operation runtime and host wakeup path | `next/tests/acceptance/test_approval_journey.py::test_approval_resumes_same_operation` | MVP | replace |
| MB-AR-04 | Completed discovery/read tasks are not replayed before the approved task. | `tests/unit/db/test_governance_runtime.py::test_resume_operation_executes_approved_blocked_task_once_and_skips_completed`; `tests/unit/db/test_agent_loop_completion_targets.py::test_resume_skips_completed_tasks_and_does_not_replay_them` | Operation runtime dependency scheduler | `next/tests/acceptance/test_approval_journey.py::test_resume_does_not_replay_completed_reads` | MVP | replace |
| MB-AR-05 | Denial is visible to both model and user. | `tests/unit/db/test_governance_runtime.py::test_rejected_expired_and_cancelled_approvals_remain_inspectable` | Governance observation and public projection | `next/tests/acceptance/test_approval_journey.py::test_denial_is_model_and_user_visible` | MVP | replace |
| MB-AR-06 | Repeated resume does not duplicate the side effect. | `tests/unit/db/test_governance_runtime.py::test_concurrent_resume_claims_side_effecting_task_once`; `tests/unit/db/test_db_monitor_scheduler.py::test_approved_governed_delivery_resumes_and_sends_once` | Operation runtime idempotency and recovery owner | `next/tests/acceptance/test_approval_journey.py::test_repeated_resume_commits_side_effect_once` | MVP | replace |
| MB-AR-07 | Approval decision racing cancellation has one deterministic outcome. | `tests/unit/runtime/test_kernel.py` approval/cancellation cases; Section 17.3 race requirement | Approval service and operation terminal transition | `next/tests/contract/runtime/test_approval_races.py::test_approval_cancel_race_has_one_terminal_outcome` | MVP | replace |

### Catalog and resource truth

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-CR-01 | Normalized resource IDs are stable for a source and native identity. | `tests/unit/catalog/test_catalog_normalizer.py`; `daita/plugins/catalog/` | Catalog identity service | `next/tests/contract/catalog/test_resource_identity.py::test_identity_is_stable_across_sync` | MVP | port |
| MB-CR-02 | Revisions and freshness distinguish current from stale structure. | PostgreSQL profile/freshness tests; `daita/plugins/catalog/` | Catalog revision and sync service | `next/tests/contract/catalog/test_revisions.py::test_revision_and_freshness_mark_stale_structure` | MVP | replace |
| MB-CR-03 | Declared and inferred relationships retain provenance and confidence. | `tests/unit/catalog/test_catalog_relationships.py`; `tests/unit/catalog/test_catalog_discoverer.py` | Catalog relationship store | `next/tests/contract/catalog/test_relationships.py::test_edges_retain_kind_provenance_and_confidence` | MVP | replace |
| MB-CR-04 | Catalog search and bounded traversal find join-relevant resources. | `tests/unit/catalog/test_catalog_relationships.py::test_catalog_find_join_path_returns_sql_ready_predicates` | Catalog search and graph traversal | `next/tests/acceptance/test_catalog_query.py::test_search_and_bounded_path_find_join_resources` | MVP | replace |
| MB-CR-05 | Profile, sample, and value facts remain distinct from structural facts. | PostgreSQL catalog profile/value tests; `tests/unit/db/test_context_projection.py` | Catalog facets and projection builder | `next/tests/contract/catalog/test_facets.py::test_observed_facts_do_not_mutate_structure` | MVP | replace |
| MB-CR-06 | Sensitive classifications affect model/public projections. | `tests/unit/db/test_context_projection.py::test_public_diagnostic_and_audit_evidence_projection_modes` | Catalog classification plus projection policy | `next/tests/contract/projections/test_sensitive_catalog.py::test_sensitive_facts_are_redacted_by_projection` | MVP | replace |
| MB-CR-07 | The data domain consumes catalog facts rather than rediscovering a private schema model. | `tests/unit/db/test_agent_loop_phase2.py::test_propose_sql_read_inserts_catalog_search_and_asset_prerequisites`; catalog ownership tests | Catalog service and data domain controller | `next/tests/architecture/test_dependency_boundaries.py::test_data_domain_has_no_private_schema_owner` | MVP | replace |

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
| MB-MO-01 | Create, list, inspect, pause, resume, run-now, and delete lifecycle is durable. | `tests/unit/db/test_db_monitors.py::test_db_agent_typed_monitor_crud_records_runtime_operations`; `tests/unit/db/test_db_monitor_commands.py` | Monitor service and monitor repository | `next/tests/acceptance/test_monitor_lifecycle.py::test_crud_pause_resume_run_now_survive_restart` | MVP | replace |
| MB-MO-02 | Two schedulers cannot claim the same due tick. | `tests/unit/db/test_db_monitor_scheduler.py::test_two_schedulers_share_one_lease_and_only_one_triggers`; live tick-lease test | Monitor scheduler and fenced tick repository | `next/tests/contract/monitors/test_tick_leases.py::test_two_schedulers_create_one_occurrence` | MVP | replace |
| MB-MO-03 | Checkpoint/cursor progress commits with the monitor outcome. | `tests/unit/db/test_db_monitor_scheduler.py::test_run_commit_rejects_stale_monitor_or_state_snapshot` | Monitor repository atomic outcome commit | `next/tests/contract/monitors/test_checkpoints.py::test_cursor_and_outcome_commit_atomically` | MVP | replace |
| MB-MO-04 | Cooldown, backoff, and missed-run policy survive restart. | `tests/unit/db/test_db_monitor_scheduler.py::test_scheduler_respects_pause_cooldown_and_backoff_gates`; live cooldown durability test | Monitor scheduler | `next/tests/acceptance/test_monitor_recovery.py::test_schedule_gates_and_catch_up_once_survive_restart` | MVP | replace |
| MB-MO-05 | One due occurrence creates at most one ordinary operation. | `tests/unit/db/test_db_monitor_scheduler.py::test_triggered_tick_creates_generic_operation_and_counts_consecutive_matches`; repeated live scheduler tick test | Trigger inbox, monitor scheduler, and operation service | `next/tests/acceptance/test_monitor_journey.py::test_due_occurrence_creates_one_normal_operation` | MVP | replace |
| MB-MO-06 | Monitor-triggered writes use ordinary policy and approval. | `tests/unit/db/test_db_monitor_scheduler.py::test_monitor_write_governance_uses_task_from_plan_task_specs`; live governed-write monitor test | Monitor service plus ordinary operation runtime | `next/tests/acceptance/test_monitor_journey.py::test_monitor_side_effect_uses_normal_approval` | MVP | replace |
| MB-MO-07 | A monitor never bypasses loop readiness, task evidence, or operation audit. | `tests/unit/db/test_db_monitor_scheduler.py::test_monitor_scheduler_does_not_bypass_runtime_execution_boundaries`; live observation contract | Generic loop and operation runtime; monitor only emits triggers | `next/tests/architecture/test_monitor_boundaries.py::test_monitor_has_no_direct_execution_path` | MVP | replace |

### Sessions, providers, and projections

| Stable ID | Behavior / feature | v1 reference | Intended v2 owner | Planned v2 black-box / contract test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- | --- |
| MB-SP-01 | Stateful follow-up uses only the intended session context. | `tests/unit/db/test_session_context.py`; `tests/integration/from_db/test_from_db_live_production_contracts.py::test_live_stateful_followup_uses_session_context` | Session service and context builder | `next/tests/acceptance/test_sessions.py::test_follow_up_uses_selected_session_only` | MVP | replace |
| MB-SP-02 | Concurrent sessions cannot leak messages, scope, approvals, or evidence. | `tests/unit/db/test_session_context.py`; live stateless non-leak test | Session repository, context builder, and projection policy | `next/tests/contract/sessions/test_isolation.py::test_concurrent_sessions_do_not_cross_contaminate` | MVP | replace |
| MB-SP-03 | Every advertised provider normalizes messages, tool calls, streaming deltas, stop reasons, usage, and retryable errors to canonical types. | `tests/unit/llm/`; `tests/integration/llm/test_llm_providers_live.py` | Model provider adapters and provider conformance suite | `next/tests/contract/models/test_provider_conformance.py::test_advertised_providers_satisfy_canonical_contract` | cutover | port |
| MB-SP-04 | Provider switching continues from canonical state; provider-native transcripts are not authoritative. | provider live tests; v1 canonical message translation in `daita/llm/` | Model router and canonical session/operation stores | `next/tests/acceptance/test_provider_continuity.py::test_fallback_continues_from_canonical_state` | cutover | replace |
| MB-SP-05 | Missing optional SDKs preserve minimal import and report the documented install hint when selected. | `tests/unit/llm/test_provider_lifecycle.py`; plugin lazy-import tests | Packaging plus lazy provider/adapter factories | `next/tests/contract/packaging/test_optional_dependencies.py::test_minimal_import_and_selection_hints` | cutover | port |
| MB-SP-06 | Audit, model, and public projections differ according to sensitivity and retention rules. | `tests/unit/db/test_context_projection.py`; `tests/unit/db/test_result_projection.py` | Projection policy and context builder | `next/tests/contract/projections/test_projection_boundaries.py::test_audit_model_public_views_are_distinct` | MVP | port |

## Root public-export dispositions

Every name in root `daita.__all__`, `daita.db.__all__`, and
`daita.llm.__all__` at the v1 baseline appears in this section. Grouping does
not imply one v2 class per v1 name; v2 keeps the behavior only where the target
architecture assigns a coherent owner.

| Stable ID | v1 public names / feature | Intended v2 owner and target surface | Planned v2 test | Classification | Final disposition |
| --- | --- | --- | --- | --- | --- |
| API-ROOT-01 | `Agent` | Thin persistent `Agent.create/open/attach/run/stream/inspect/approve/reject/cancel/resume` facade | `next/tests/acceptance/test_public_agent.py` | MVP | replace |
| API-ROOT-02 | `BaseAgent` | Composition through `Agent`, domain-controller, and service protocols; no subclass framework | `next/tests/architecture/test_public_surface.py::test_no_parallel_base_agent_framework` | cutover | proposed removal requiring Phase 10 approval |
| API-ROOT-03 | `ConversationHistory` | Session/message service and canonical operation transcript | `next/tests/acceptance/test_sessions.py` | MVP | replace |
| API-ROOT-04 | `tool` | Explicit local capability declaration projected as a `ToolView`; never direct loop execution | `next/tests/contract/extensions/test_local_capability.py` | cutover | replace |
| API-ROOT-05 | `configure_tracing`, `get_trace_manager`, `set_trace_context` | Event-to-telemetry projection and optional exporter registration | `next/tests/contract/telemetry/test_event_projection.py` | cutover | replace |
| API-ROOT-06 | `postgresql`, `sqlite` | Built-in resource adapters and declared executors | Shared SQLite/PostgreSQL adapter conformance suite | cutover | replace |
| API-ROOT-07 | `mysql`, `mongodb`, `rest`, `s3`, `slack`, `elasticsearch`, `redis_messaging` | Future explicit resource/capability adapters; absent from candidate runtime | Phase 9 absence/install-hint and support-matrix checks | post-MVP | defer (documented) |
| API-ROOT-08 | `BasePlugin`, `ConnectorPlugin`, `DomainServicePlugin`, `ObservabilityPlugin`, `RuntimeExtensionPlugin`, `SkillPlugin`, `WorkerProviderPlugin` | Narrow resource-adapter, capability-provider, backend-provider, manifest, and observer protocols | `next/tests/architecture/test_extension_boundaries.py` | cutover | proposed removal requiring Phase 10 approval |
| API-ROOT-09 | `EmptySecretProvider`, `SecretProvider` | Injectable secret-provider protocol plus explicit empty/env/keychain implementations | `next/tests/contract/security/test_secret_provider.py` | cutover | replace |
| API-ROOT-10 | `PluginContext`, `PluginKind`, `PluginManifest`, `ServiceRegistry`, `ExtensionRegistry`, `RegistryDiagnostic` | Validated narrow extension manifest/registry and diagnostics | `next/tests/contract/extensions/test_registry.py` | cutover | replace |
| API-ROOT-11 | `BaseSkill`, `Skill`, `SkillActivation`, `SkillActivationRules`, `SkillDiscovery`, `SkillResolver`, `SkillResolution` | Versioned `SKILL.md` index, selection, activation, and service records | `next/tests/unit/skills/test_skill_service.py`; `next/tests/acceptance/test_skill_change_lifecycle.py::test_public_skill_change_propose_accept_reopen_lifecycle` | MVP | replace |
| API-ROOT-12 | `SkillRuntimeEffects` | Skills may reference capabilities but cannot declare runtime effects/executors | `next/tests/unit/skills/test_skill_boundaries.py`; `next/tests/architecture/test_phase5_context_learning_architecture.py` | cutover | proposed removal requiring Phase 10 approval |
| API-ROOT-13 | `AgentConfig`, `RetryPolicy`, `RetryStrategy` | Versioned agent/model/budget/policy configuration owned by services and router | `next/tests/contract/config/test_agent_configuration.py` | cutover | replace |
| API-ROOT-14 | `apply_focus` | Bounded capability arguments and evidence projections; the standalone v1 Focus DSL is not an MVP owner | Phase 9 migration/support-matrix assertion | post-MVP | proposed removal requiring Phase 10 approval |
| API-ROOT-15 | `DaitaError`, `AgentError`, `LLMError`, `ConfigError`, `PluginError`, `SkillError`, `TransientError`, `RetryableError`, `PermanentError`, `RateLimitError`, `AuthenticationError`, `ValidationError`, `FocusDSLError`, `DataQualityError` | Stable typed public result/error taxonomy, normalized provider errors, and subsystem-specific details | `next/tests/contract/test_errors.py` | cutover | replace |
| API-ROOT-16 | `ItemAssertion` | Future data-quality capability provider; not part of architecture MVP | Phase 9 documented-absence assertion | post-MVP | defer (documented) |
| API-ROOT-17 | `create_llm_provider` | Model registry/router configured by canonical provider/model profiles | `next/tests/contract/models/test_registry.py` | MVP | replace |
| API-ROOT-18 | `BaseEmbeddingProvider` | Separate embedding-provider protocol; lexical FTS is authoritative for MVP | Phase 9 documented-absence assertion | post-MVP | defer (documented) |
| API-ROOT-19 | `__version__` | Package version metadata | `next/tests/contract/packaging/test_version.py` | cutover | port |
| API-DB-01 | `from_db`, `DbAgent` | `Agent.create/open`, `attach(SQLiteSource/PostgreSQLSource)`, then the same `Agent.run`; no DB agent subclass | `next/tests/acceptance/test_grounded_query.py` | cutover | replace |
| API-DB-02 | `DbRuntime` | Generic operation runtime plus built-in data-domain controller | `next/tests/architecture/test_dependency_boundaries.py` | MVP | proposed removal requiring Phase 10 approval |
| API-DB-03 | `DbIntent`, `DbIntentKind`, `DbRequest`, `DbOperationContract`, `DbOperationResult` | Canonical trigger/operation/action/readiness/exit and public result records | `next/tests/contract/loop/test_models.py` | MVP | replace |
| API-DB-04 | `DbLimits`, `DbRuntimeConfig`, `DbRuntimeOptions`, `DbExecutionConfig` | Agent, loop-budget, operation-runtime, store, and host configuration | `next/tests/contract/config/test_runtime_configuration.py` | cutover | replace |
| API-DB-05 | `DbRuntimeInspection` | Generic operation/agent inspection views | `next/tests/acceptance/test_operation_inspection.py` | MVP | replace |
| API-DB-06 | `DbSourceOptions` | Typed SQLite/PostgreSQL source configuration and scope policy | Adapter conformance source-scope cases | cutover | replace |
| API-DB-07 | `DbLLMConfig` | Canonical model profiles and router policy | `next/tests/contract/models/test_router.py` | cutover | replace |
| API-DB-08 | `DbMemoryConfig` | Memory service policy/configuration | `next/tests/contract/memory/test_configuration.py` | MVP | replace |
| API-DB-09 | `DbMonitor`, `DbMonitorInspection`, `DbMonitorMutation`, `DbMonitorRun`, `DbMonitorState`, `DbMonitorStore` | Generic monitor service, records, repository, scheduler, and inspection API | `next/tests/acceptance/test_monitor_lifecycle.py` | MVP | replace |
| API-DB-10 | `HostedInAppMonitorDeliveryPlugin` | Local durable finding/event; outbound delivery is a later explicit extension | Phase 9 documented-absence assertion | post-MVP | defer (documented) |
| API-LLM-01 | `register_llm_provider`, `list_available_providers`, `BaseLLMProvider` | Canonical model-provider protocol and explicit registry | `next/tests/contract/models/test_registry.py` | cutover | replace |
| API-LLM-02 | `CostEstimate`, `ModelPricing`, `TokenUsage`, `estimate_llm_cost` | Canonical usage/cost records and router budget estimator | `next/tests/contract/models/test_usage.py` | cutover | port |
| API-LLM-03 | `OpenAIProvider`, `AnthropicProvider`, `GrokProvider`, `GeminiProvider`, `MockLLMProvider`; factory-only `OllamaProvider`; supported OpenAI-compatible endpoints | Lazy model adapters satisfying the shared provider conformance suite | `next/tests/contract/models/test_provider_conformance.py` plus live acceptance markers | cutover | port |

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
| EXT-01 | `dev` | V2-only test/build/type/format dependencies | MVP | replace |
| EXT-02 | `sqlite` | Required architecture-MVP adapter and data-domain conformance | MVP | replace |
| EXT-03 | `postgresql` | Required replacement-candidate adapter through the same contracts | cutover | replace |
| EXT-04 | `anthropic`, `google`, `llm-all` | Retained model adapters; `google` maps to Gemini | cutover | replace |
| EXT-05 | `transformers` | V1 local Transformers model path is not the retained local-provider contract; Ollama/OpenAI-compatible is | post-MVP | defer (documented) |
| EXT-06 | `data` | Phase 4 completes the v2-native bounded, sandboxed CSV/JSON path without pandas; remaining cutover work is data-extra/API parity, with XLSX optional only if inexpensive | cutover | replace |
| EXT-07 | `memory` | Memory is a core lifecycle using SQLite FTS; embedding/graph packages are not required semantics | MVP | replace |
| EXT-08 | `sentence-transformers`, `voyage` | Optional embedding accelerators | post-MVP | defer (documented) |
| EXT-09 | `data-quality`, `lineage` | Future capability providers, not alternate runtimes | post-MVP | defer (documented) |
| EXT-10 | `mysql`, `mongodb`, `snowflake`, `bigquery`, `elasticsearch`, `opensearch`, `databases` | Additional databases/search sources after SQLite/PostgreSQL contracts are stable | post-MVP | defer (documented) |
| EXT-11 | `chromadb`, `pinecone`, `qdrant`, `vectordb`, `neo4j` | Vector/graph acceleration is not required for catalog or memory semantics | post-MVP | defer (documented) |
| EXT-12 | `redis` | Distributed state/messaging is not part of the local one-writer MVP | post-MVP | defer (documented) |
| EXT-13 | `aws`, `azure`, `gcp`, `google-drive`, `cloud` | Cloud/object/document resource adapters are on the post-MVP roadmap | post-MVP | defer (documented) |
| EXT-14 | `github`, `slack`, `mcp` | App/communication/protocol integrations are explicit later adapters/delivery extensions | post-MVP | defer (documented) |
| EXT-15 | `web`, `websearch`, `exa` | Web content/search adapters are outside the data-source MVP | post-MVP | defer (documented) |
| EXT-16 | `cli` | External CLI parsing may be retained, but Phase 6 ships a thin development CLI against `AgentHost` and Phase 9 integrates the production package | cutover | external integration |
| EXT-17 | `api-server`, `production` | Local host API/serve behavior is ported; generic v1 deployment bundles are not candidate runtime dependencies | cutover | replace |
| EXT-18 | `otlp` | Optional exporter consumes committed canonical events/spans after local tracing is stable | cutover | replace |
| EXT-19 | `recommended`, `complete`, `all` | Rebuild only from supported v2 extras; release artifacts must document omitted v1 members | cutover | replace |

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
| SURF-01 | Examples `00_quickstart_sqlite_from_db.py`, `01_inspectable_operation.py`, `02_catalog_assisted_joins.py`, `03_governed_reads_and_writes.py`, `04_persistent_runtime_store.py`, `06_memory_for_business_semantics.py`, `07_monitor_orders.py`, `09_custom_data_plugin_extension.py`, `10_csv_to_sqlite_data_app.py` | Rewrite against v2 public API/host contracts and new evidence model | cutover | replace |
| SURF-02 | Examples `05_data_quality_and_lineage.py`, `08_infrastructure_catalog.py` | Document as deferred with their capability/resource adapters | post-MVP | defer (documented) |
| SURF-03 | `examples/deployments/data-team-agent/` | Rebuild as Phase 9 production-shaped local-host example | cutover | replace |
| SURF-04 | `daita.evals` developer-preview API and structured eval artifacts | Keep neutral datasets/fixtures where useful; v1 eval framework is not a candidate runtime dependency | post-MVP | defer (documented) |
| SURF-05 | Generic local tool agents, streaming, and detailed diagnostics | Single loop, explicit capabilities, canonical committed events, and public inspection results | cutover | replace |
| SURF-06 | Direct plugin context managers such as `async with sqlite(...)` | Explicit adapter lifecycle remains available only if it does not bypass the operation runtime for agent-owned work | cutover | replace |
| SURF-07 | External `daita-cli` and `daita-client` packages | Integrate through proven Phase 6 host/SDK contracts; never own local runtime semantics | cutover | external integration |

## Completion accounting

- Mandatory behavior rows: **51 of 51 classified**.
- Root `daita.__all__`: **61 of 61 names dispositioned**.
- Root `daita.db.__all__`: **23 of 23 names dispositioned**.
- Root `daita.llm.__all__`: **13 of 13 names dispositioned** (including the
  duplicate root LLM factory by cross-reference).
- Root optional extras: **44 of 44 names dispositioned**.
- Test-file-level classification is maintained separately in
  `TEST_DISPOSITION.csv` and validated against Git-tracked root tests.
- A disposition is not implementation evidence. Each port/replace row remains
  open until its planned v2 test passes at the named gate.
