BEGIN TRANSACTION;
CREATE TABLE agent_home_migrations (
    revision INTEGER NOT NULL PRIMARY KEY,
    migration_id TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL
);
INSERT INTO "agent_home_migrations" VALUES(1,'agent_home_revision_1','a08bdc56e3cb7c3dbe77dc0d8b8ed9aac1299a6a701902dba8e11a5ebe70f25e');
INSERT INTO "agent_home_migrations" VALUES(2,'0002_durable_adaptive_task_graph','cb98e978fcb6db71cc1cb3ee5532d92ce99117db7359a008a98a4651df1c20e2');
CREATE TABLE deliveries (
    agent_id TEXT NOT NULL,
    delivery_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_fingerprint TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, delivery_id),
    UNIQUE (agent_id, logical_key),
    UNIQUE (agent_id, subject_kind, subject_id, target_fingerprint)
);
CREATE TABLE effect_receipts (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    routine_id TEXT,
    occurrence_id TEXT,
    grant_digest TEXT,
    unresolved INTEGER NOT NULL,
    job_id TEXT,
    task_id TEXT,
    task_attempt_id TEXT,
    fencing_epoch INTEGER,
    task_spec_digest TEXT,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id),
    UNIQUE(agent_id, run_id, call_id),
    UNIQUE(agent_id, operation_key),
    CHECK (
        (job_id IS NULL AND task_id IS NULL AND task_attempt_id IS NULL
         AND fencing_epoch IS NULL AND task_spec_digest IS NULL)
        OR
        (job_id IS NOT NULL AND task_id IS NOT NULL AND task_attempt_id IS NOT NULL
         AND fencing_epoch IS NOT NULL AND task_spec_digest IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id, task_id, task_attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE TABLE job_attempt_budget_reservations (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    reserved INTEGER NOT NULL CHECK (reserved >= 0),
    settled INTEGER CHECK (settled IS NULL OR settled >= 0),
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY(agent_id, job_id, task_id, attempt_id, dimension),
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE TABLE job_budget_ledger (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    ceiling INTEGER NOT NULL CHECK (ceiling >= 0),
    settled INTEGER NOT NULL CHECK (settled >= 0),
    reserved INTEGER NOT NULL CHECK (reserved >= 0),
    control_reserved INTEGER NOT NULL CHECK (control_reserved >= 0),
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY(agent_id, job_id, dimension),
    CHECK (settled + reserved <= ceiling),
    CHECK (control_reserved <= ceiling),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT
);
CREATE TABLE job_graph_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT,
    attempt_id TEXT,
    kind TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    CHECK (attempt_id IS NULL OR task_id IS NOT NULL),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE TABLE job_graph_mutations (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    mutation_id TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    actor_key TEXT NOT NULL,
    creator_task_id TEXT,
    creator_attempt_id TEXT,
    idempotency_key TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    expected_revision INTEGER NOT NULL CHECK (expected_revision >= 0),
    committed_revision INTEGER NOT NULL CHECK (committed_revision >= 0),
    decision TEXT NOT NULL CHECK (decision IN ('committed','rejected')),
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, mutation_id),
    UNIQUE(agent_id, job_id, actor_key, idempotency_key),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, creator_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, creator_task_id, creator_attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE TABLE job_graphs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    task_count INTEGER NOT NULL CHECK (task_count >= 0),
    edge_count INTEGER NOT NULL CHECK (edge_count >= 0),
    mutation_count INTEGER NOT NULL CHECK (mutation_count >= 0),
    active_attempt_count INTEGER NOT NULL CHECK (active_attempt_count >= 0),
    next_ready_at_us INTEGER,
    finalization_attempt_id TEXT,
    finalization_started_revision INTEGER,
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id),
    CHECK (
        (finalization_attempt_id IS NULL AND finalization_started_revision IS NULL)
        OR
        (finalization_attempt_id IS NOT NULL
         AND finalization_started_revision IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_runs(agent_id, job_id) ON DELETE RESTRICT
);
CREATE TABLE job_runs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    origin_run_id TEXT NOT NULL,
    origin_call_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'queued','active','blocked','needs_attention','cancel_requested',
        'succeeded','failed','cancelled'
    )),
    desired_state TEXT NOT NULL CHECK (desired_state IN ('run','cancel')),
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    deadline_at_us INTEGER NOT NULL,
    terminal_at_us INTEGER,
    finalizer_task_id TEXT,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id),
    UNIQUE(agent_id, origin_run_id, origin_call_id)
);
CREATE TABLE job_task_attempts (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    fencing_epoch INTEGER NOT NULL CHECK (fencing_epoch > 0),
    state TEXT NOT NULL CHECK (state IN (
        'claimed','running','succeeded','failed','cancelled','blocked',
        'review_requested','timed_out','protocol_violation','fenced'
    )),
    claim_token TEXT NOT NULL,
    run_id TEXT NOT NULL,
    lease_expires_at_us INTEGER,
    absolute_deadline_at_us INTEGER NOT NULL,
    started_at_us INTEGER,
    heartbeat_at_us INTEGER,
    ended_at_us INTEGER,
    active_slot INTEGER,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, attempt_id),
    UNIQUE(agent_id, job_id, task_id, ordinal),
    UNIQUE(agent_id, job_id, task_id, fencing_epoch),
    UNIQUE(agent_id, run_id),
    UNIQUE(agent_id, job_id, task_id, active_slot),
    CHECK (
        (state IN ('claimed','running') AND active_slot = 1
         AND lease_expires_at_us IS NOT NULL AND ended_at_us IS NULL)
        OR
        (state NOT IN ('claimed','running') AND active_slot IS NULL
         AND lease_expires_at_us IS NULL AND ended_at_us IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE TABLE job_task_budget_ledger (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    ceiling INTEGER NOT NULL CHECK (ceiling >= 0),
    settled INTEGER NOT NULL CHECK (settled >= 0),
    reserved INTEGER NOT NULL CHECK (reserved >= 0),
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY(agent_id, job_id, task_id, dimension),
    CHECK (settled + reserved <= ceiling),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE TABLE job_task_checkpoints (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    created_at_us INTEGER NOT NULL,
    payload_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, attempt_id, checkpoint_id),
    UNIQUE(agent_id, job_id, task_id, attempt_id, ordinal),
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE TABLE job_task_comments (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    comment_id TEXT NOT NULL,
    author_kind TEXT NOT NULL,
    author_id TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    body_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, comment_id),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE TABLE job_task_controls (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    control_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN (
        'needs_input','needs_authorization','needs_replan','review_requested',
        'changes_requested','effect_uncertain','capability_unavailable',
        'source_or_contract_drift','budget_exhausted','retry_circuit_open'
    )),
    state TEXT NOT NULL CHECK (state IN ('open','resolved','rejected','expired')),
    requesting_attempt_id TEXT,
    created_at_us INTEGER NOT NULL,
    resolved_at_us INTEGER,
    resolved_by_kind TEXT,
    resolved_by_id TEXT,
    payload_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, control_id),
    CHECK (
        (state = 'open' AND resolved_at_us IS NULL
         AND resolved_by_kind IS NULL AND resolved_by_id IS NULL)
        OR
        (state <> 'open' AND resolved_at_us IS NOT NULL
         AND resolved_by_kind IS NOT NULL AND resolved_by_id IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id, requesting_attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE TABLE job_task_dependencies (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    upstream_task_id TEXT NOT NULL,
    downstream_task_id TEXT NOT NULL,
    edge_kind TEXT NOT NULL CHECK (edge_kind = 'requires_accepted_success'),
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, upstream_task_id, downstream_task_id),
    CHECK (upstream_task_id <> downstream_task_id),
    FOREIGN KEY(agent_id, job_id, upstream_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, downstream_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE TABLE job_task_results (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    result_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    completed_at_us INTEGER NOT NULL,
    sensitivity TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id),
    UNIQUE(agent_id, job_id, result_id),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE TABLE job_tasks (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'pending','ready','running','blocked','review','succeeded','failed',
        'cancelled','skipped','superseded'
    )),
    role TEXT NOT NULL CHECK (role IN (
        'planner','worker','reviewer','finalizer','internal'
    )),
    task_kind TEXT NOT NULL CHECK (task_kind IN ('model','internal_capability')),
    priority INTEGER NOT NULL,
    not_before_us INTEGER,
    current_attempt_id TEXT,
    task_revision INTEGER NOT NULL CHECK (task_revision > 0),
    task_spec_digest TEXT NOT NULL,
    task_scope_digest TEXT NOT NULL,
    supersedes_task_id TEXT,
    superseded_by_task_id TEXT,
    latest_result_id TEXT,
    latest_control_id TEXT,
    latest_checkpoint_id TEXT,
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    terminal_at_us INTEGER,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id),
    CHECK (supersedes_task_id IS NULL OR supersedes_task_id <> task_id),
    CHECK (superseded_by_task_id IS NULL OR superseded_by_task_id <> task_id),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, supersedes_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, superseded_by_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE TABLE learning_candidates (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE TABLE mcp_server_bindings (
    agent_id TEXT NOT NULL,
    binding_id TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, binding_id)
);
CREATE TABLE messages (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(run_id, position)
);
INSERT INTO "messages" VALUES('run-golden-revision-1',0,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"TextBlock","fields":{"text":"Golden question."}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"user"},"tool_calls":[]}}');
INSERT INTO "messages" VALUES('run-golden-revision-1',1,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"TextBlock","fields":{"text":"Golden answer."}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"assistant"},"tool_calls":[]}}');
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
INSERT INTO "metadata" VALUES('identity','{"__record__":"AgentIdentity","fields":{"created_at":{"__datetime__":"2026-01-02T03:04:05Z"},"display_name":"golden","id":"agent-golden-revision-1"}}');
CREATE TABLE relational_write_scopes (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    authorization_fingerprint TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, source_id, resource_id),
    FOREIGN KEY (agent_id, source_id)
        REFERENCES sources(agent_id, id)
        ON DELETE CASCADE
);
CREATE TABLE routine_occurrences (
    agent_id TEXT NOT NULL,
    occurrence_id TEXT NOT NULL,
    routine_id TEXT NOT NULL,
    routine_revision INTEGER NOT NULL,
    slot_key TEXT NOT NULL,
    state TEXT NOT NULL,
    lease_expires_at_us INTEGER,
    reserved_run_id TEXT,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, occurrence_id),
    UNIQUE (agent_id, routine_id, routine_revision, slot_key),
    UNIQUE (agent_id, reserved_run_id),
    FOREIGN KEY (agent_id, routine_id)
        REFERENCES scheduled_routines(agent_id, routine_id)
        ON DELETE CASCADE
);
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    input TEXT NOT NULL,
    result TEXT
);
INSERT INTO "runs" VALUES('run-golden-revision-1','agent-golden-revision-1','conversation-golden-revision-1',0,'{"__record__":"RunInput","fields":{"agent_id":"agent-golden-revision-1","conversation_id":"conversation-golden-revision-1","created_at":{"__datetime__":"2026-01-02T03:04:05Z"},"history_sensitivity":"public","id":"run-golden-revision-1","message":"Golden question.","resolved_source_scope":null,"source_scope_ids":[],"start":{"__record__":"RunStartEnvelope","fields":{"execution_scope":null,"instruction_authority":null,"instruction_digest":null,"origin":"user","payload_digest":null,"trusted_instruction":null,"trusted_instruction_id":null,"untrusted_payload":{},"user_message":"Golden question."}}}}','{"__record__":"LoopExit","fields":{"artifact_deliveries":[],"artifacts":[],"conversation_id":"conversation-golden-revision-1","created_at":{"__datetime__":"2026-01-02T03:04:05Z"},"final_text":"Golden answer.","kind":{"__enum__":"LoopExitKind","value":"completed"},"provider_failure":null,"provider_id":null,"reason":"completed","run_id":"run-golden-revision-1","sensitivity":"restricted","steps":0,"usage":{"__record__":"ModelUsage","fields":{"cache_read_tokens":0,"cache_write_tokens":0,"cost_estimate":{"__record__":"CostEstimate","fields":{"amount_usd":null,"basis":null,"code":"pricing_schedule_unavailable","components":[],"rate_schedule_id":null,"status":{"__enum__":"CostEstimateStatus","value":"unavailable"}}},"input_tokens":0,"output_tokens":0,"reasoning_tokens":0}}}}');
CREATE TABLE scheduled_routines (
    agent_id TEXT NOT NULL,
    routine_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    state TEXT NOT NULL,
    next_due_at_us INTEGER,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, routine_id)
);
CREATE TABLE semantic_annotations (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE TABLE snapshots (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    sync_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, source_id)
);
CREATE TABLE source_read_scopes (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, source_id),
    FOREIGN KEY (agent_id, source_id)
        REFERENCES sources(agent_id, id)
        ON DELETE CASCADE
);
CREATE TABLE sources (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE TABLE syncs (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE UNIQUE INDEX runs_conversation_turn
    ON runs(agent_id, conversation_id, turn_index);
CREATE INDEX effect_receipts_unresolved
    ON effect_receipts(agent_id, unresolved, routine_id, run_id);
CREATE INDEX effect_receipts_grant_reservations
    ON effect_receipts(agent_id, occurrence_id, grant_digest);
CREATE INDEX effect_receipts_graph_attempt
    ON effect_receipts(agent_id, job_id, task_id, task_attempt_id)
;
CREATE INDEX job_runs_state_deadline
    ON job_runs(agent_id, state, deadline_at_us);
CREATE INDEX job_runs_conversation_created
    ON job_runs(agent_id, conversation_id, created_at_us);
CREATE INDEX job_runs_updated
    ON job_runs(agent_id, updated_at_us);
CREATE INDEX job_graphs_ready
    ON job_graphs(agent_id, next_ready_at_us, updated_at_us);
CREATE INDEX job_tasks_ready
    ON job_tasks(agent_id, state, not_before_us, priority, updated_at_us);
CREATE INDEX job_tasks_by_job
    ON job_tasks(agent_id, job_id, state, priority, created_at_us);
CREATE INDEX job_tasks_current_attempt
    ON job_tasks(agent_id, job_id, current_attempt_id);
CREATE INDEX job_task_dependencies_reverse
    ON job_task_dependencies(
        agent_id, job_id, downstream_task_id, upstream_task_id
    );
CREATE INDEX job_task_attempts_stale
    ON job_task_attempts(agent_id, state, lease_expires_at_us);
CREATE INDEX job_task_attempts_by_task
    ON job_task_attempts(agent_id, job_id, task_id, ordinal);
CREATE INDEX job_task_attempts_deadline
    ON job_task_attempts(agent_id, job_id, state, absolute_deadline_at_us);
CREATE INDEX job_task_controls_open
    ON job_task_controls(agent_id, job_id, state, created_at_us);
CREATE INDEX job_graph_events_job
    ON job_graph_events(agent_id, job_id, event_id);
CREATE INDEX job_graph_events_task
    ON job_graph_events(agent_id, job_id, task_id, event_id);
CREATE INDEX job_graph_events_kind
    ON job_graph_events(agent_id, kind, event_id);
CREATE INDEX deliveries_conversation_history
    ON deliveries(agent_id, conversation_id, created_at_us, delivery_id)
;
CREATE INDEX scheduled_routines_due
    ON scheduled_routines(agent_id, state, next_due_at_us, routine_id)
;
CREATE INDEX routine_occurrences_stale
    ON routine_occurrences(agent_id, state, lease_expires_at_us, occurrence_id)
;
DELETE FROM "sqlite_sequence";
COMMIT;
