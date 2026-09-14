BEGIN TRANSACTION;
CREATE TABLE agent_home_migrations (
    revision INTEGER NOT NULL PRIMARY KEY,
    migration_id TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL
);
INSERT INTO "agent_home_migrations" VALUES(1,'agent_home_revision_1','a08bdc56e3cb7c3dbe77dc0d8b8ed9aac1299a6a701902dba8e11a5ebe70f25e');
INSERT INTO "agent_home_migrations" VALUES(2,'agent_home_revision_2','1090ba2cf009d09516f603832cba4e7224b2a33c841d21b159e597dff5e41b86');
CREATE TABLE autonomous_followups (
    agent_id TEXT NOT NULL,
    followup_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, followup_id),
    UNIQUE (agent_id, event_id),
    UNIQUE (agent_id, job_id)
);
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
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id),
    UNIQUE(agent_id, run_id, call_id),
    UNIQUE(agent_id, operation_key)
);
CREATE TABLE job_runs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, job_id)
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
INSERT INTO "runs" VALUES('run-golden-revision-1','agent-golden-revision-1','conversation-golden-revision-1',0,'{"__record__":"RunInput","fields":{"agent_id":"agent-golden-revision-1","conversation_id":"conversation-golden-revision-1","created_at":{"__datetime__":"2026-01-02T03:04:05Z"},"history_sensitivity":"public","id":"run-golden-revision-1","message":"Golden question.","resolved_source_scope":null,"source_scope_ids":[],"start":{"__record__":"RunStartEnvelope","fields":{"execution_scope":null,"instruction_authority":null,"instruction_digest":null,"origin":"user","payload_digest":null,"trusted_instruction":null,"trusted_instruction_id":null,"untrusted_payload":{},"user_message":"Golden question."}},"target_posture":"single_target"}}','{"__record__":"LoopExit","fields":{"artifact_deliveries":[],"artifacts":[],"conversation_id":"conversation-golden-revision-1","created_at":{"__datetime__":"2026-01-02T03:04:05Z"},"final_text":"Golden answer.","kind":{"__enum__":"LoopExitKind","value":"completed"},"provider_failure":null,"provider_id":null,"reason":"completed","run_id":"run-golden-revision-1","sensitivity":"restricted","steps":0,"usage":{"__record__":"ModelUsage","fields":{"cache_read_tokens":0,"cache_write_tokens":0,"cost_estimate":{"__record__":"CostEstimate","fields":{"amount_usd":null,"basis":null,"code":"pricing_schedule_unavailable","components":[],"rate_schedule_id":null,"status":{"__enum__":"CostEstimateStatus","value":"unavailable"}}},"input_tokens":0,"output_tokens":0,"reasoning_tokens":0}}}}');
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
CREATE INDEX effect_receipts_unresolved ON effect_receipts(agent_id, unresolved, routine_id, run_id);
CREATE INDEX effect_receipts_grant_reservations ON effect_receipts(agent_id, occurrence_id, grant_digest)
;
CREATE INDEX deliveries_conversation_history
    ON deliveries(agent_id, conversation_id, created_at_us, delivery_id)
;
CREATE INDEX scheduled_routines_due
    ON scheduled_routines(agent_id, state, next_due_at_us, routine_id)
;
CREATE INDEX routine_occurrences_stale
    ON routine_occurrences(agent_id, state, lease_expires_at_us, occurrence_id)
;
COMMIT;
