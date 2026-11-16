BEGIN TRANSACTION;
CREATE TABLE agent_home_migrations (
    revision INTEGER NOT NULL PRIMARY KEY,
    migration_id TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL
);
INSERT INTO "agent_home_migrations" VALUES(1,'agent_home_revision_1','a08bdc56e3cb7c3dbe77dc0d8b8ed9aac1299a6a701902dba8e11a5ebe70f25e');
INSERT INTO "agent_home_migrations" VALUES(2,'0002_durable_adaptive_task_graph','cb98e978fcb6db71cc1cb3ee5532d92ce99117db7359a008a98a4651df1c20e2');
INSERT INTO "agent_home_migrations" VALUES(3,'0003_framework_caller_authority','3357ee7ad8153e5160d06133082be89dca1e868f1529029f5b63988fae5dca42');
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY NOT NULL,
    agent_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    caller_principal_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('creating', 'ready', 'deleting')),
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL
);
INSERT INTO "artifacts" VALUES('artifact-00000000000000000000000000000001','agent-golden-revision-1','run-00000000000000000000000000000001','conversation-00000000000000000000000000000001','agent-golden-revision-1','ready',18,1790960058476070,'{"__record__":"ArtifactRecord","fields":{"agent_id":"agent-golden-revision-1","caller_principal_id":"agent-golden-revision-1","ref":{"__record__":"ArtifactRef","fields":{"artifact_id":"artifact-00000000000000000000000000000001","byte_size":18,"call_id":"create-0","capability_id":"artifact.create_document","conversation_id":"conversation-00000000000000000000000000000001","created_at":{"__datetime__":"2026-10-02T16:54:18.476070Z"},"filename":"golden-0.txt","media_type":"text/plain","provenance":{"__record__":"ArtifactProvenance","fields":{"authorship":{"__enum__":"ArtifactAuthorship","value":"model_authored_analysis"},"columns":[],"derived_from_artifact_id":null,"evidence_call_ids":[],"local_file_binding":null,"parameters_sha256":null,"resource_bindings":[],"result_binding":null,"row_count":null,"sql_fingerprint":null}},"run_id":"run-00000000000000000000000000000001","sensitivity":{"__enum__":"Sensitivity","value":"restricted"},"sha256":"sha256:641fc61003c7639d83d76af5fe200ee094fc7faeeb48a276131efc79a0e53fee"}},"state":{"__enum__":"ArtifactState","value":"ready"}}}');
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
INSERT INTO "job_budget_ledger" VALUES('agent-golden-revision-1','job-unfinished','work_units',4,0,0,1,1789560000000000);
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
INSERT INTO "job_graph_events" VALUES(1,'agent-golden-revision-1','job-unfinished',NULL,NULL,'graph_admitted',1789560000000000,'{"__record__":"GraphEventPayload","fields":{"kind":"graph_admitted","payload":{"edge_count":1,"specification_digest":"sha256:e02fffe6b9cd8111cd4fa97fccb620fd7f018b1256a643c27aaa751eada2f0a8","task_count":2}}}');
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
INSERT INTO "job_graphs" VALUES('agent-golden-revision-1','job-unfinished',0,2,1,0,0,1789560000000000,NULL,NULL,1789560000000000,1789560000000000,'{"__record__":"JobGraph","fields":{"active_attempt_count":0,"agent_id":"agent-golden-revision-1","created_at":{"__datetime__":"2026-09-16T12:00:00Z"},"edge_count":1,"finalization_attempt_id":null,"finalization_started_revision":null,"job_id":"job-unfinished","mutation_count":0,"next_ready_at":{"__datetime__":"2026-09-16T12:00:00Z"},"revision":0,"task_count":2,"topology_digest":"sha256:aa4c4820adab9acd1c247d04a3c630c91522402b85d7ea1cf686e7a3d676339d","updated_at":{"__datetime__":"2026-09-16T12:00:00Z"}}}');
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
INSERT INTO "job_runs" VALUES('agent-golden-revision-1','job-unfinished','conversation-1','job-unfinished:origin-run','job-unfinished:origin-call','queued','run',1789560000000000,1789560000000000,1789563600000000,NULL,'finalizer','{"__record__":"GraphJob","fields":{"agent_id":"agent-golden-revision-1","conversation_id":"conversation-1","created_at":{"__datetime__":"2026-09-16T12:00:00Z"},"deadline_at":{"__datetime__":"2026-09-16T13:00:00Z"},"desired_state":"run","failure_code":null,"finalizer_task_id":"finalizer","job_id":"job-unfinished","migration_provenance":{},"origin_call_id":"job-unfinished:origin-call","origin_run_id":"job-unfinished:origin-run","specification":{"__record__":"GraphJobSpecification","fields":{"authority":{"__record__":"GraphAuthority","fields":{"access_modes":["read"],"capability_ids":["capability.read"],"connector_ids":[],"contract_bindings":{"capability.read":"sha256:1111111111111111111111111111111111111111111111111111111111111111"},"model_route_ids":[],"operational_effects":["none"],"resource_ids":[],"sensitivity":"restricted","source_ids":[]}},"budgets":[{"__record__":"BudgetLimit","fields":{"ceiling":4,"control_reserved":1,"dimension":"work_units"}}],"cancellation_policy":{"preserve_evidence":true},"deadline_at":{"__datetime__":"2026-09-16T13:00:00Z"},"distribution_plan_digest":"sha256:a1d86a3f62fea42a2d263a1745c6513695036fd1d7c54733c02730eb0f02aa37","effect_mode":"disabled","finalizer_task_template":{"kind":"promote"},"limits":{"__record__":"GraphLimits","fields":{"max_artifact_bytes":8388608,"max_artifacts":64,"max_attempts_per_task":3,"max_checkpoints_per_attempt":8,"max_comments_per_task":32,"max_controls_per_task":16,"max_depth":12,"max_direct_parents":16,"max_edges":192,"max_events":4096,"max_fan_out":16,"max_mutations":256,"max_parallelism":4,"max_tasks":64,"max_total_artifact_bytes":67108864}},"objective":"Complete the deterministic graph test.","outcome_contract":{"kind":"test"},"planner_task_template":null,"principal_id":"agent-golden-revision-1","retry_policy":{"max_attempts":3}}},"specification_digest":"sha256:e02fffe6b9cd8111cd4fa97fccb620fd7f018b1256a643c27aaa751eada2f0a8","state":"queued","terminal_at":null,"terminal_result_id":null,"updated_at":{"__datetime__":"2026-09-16T12:00:00Z"}}}');
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
INSERT INTO "job_task_budget_ledger" VALUES('agent-golden-revision-1','job-unfinished','worker','work_units',3,0,0,1789560000000000);
INSERT INTO "job_task_budget_ledger" VALUES('agent-golden-revision-1','job-unfinished','finalizer','work_units',1,0,0,1789560000000000);
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
INSERT INTO "job_task_dependencies" VALUES('agent-golden-revision-1','job-unfinished','worker','finalizer','requires_accepted_success',1789560000000000,'{"__record__":"TaskDependency","fields":{"agent_id":"agent-golden-revision-1","created_at":{"__datetime__":"2026-09-16T12:00:00Z"},"creator_key":"owner","downstream_task_id":"finalizer","edge_kind":"requires_accepted_success","job_id":"job-unfinished","mutation_id":null,"upstream_task_id":"worker"}}');
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
INSERT INTO "job_tasks" VALUES('agent-golden-revision-1','job-unfinished','worker','ready','worker','model',10,NULL,NULL,1,'sha256:e5d8cb59770ee00af6f86aeae6f43f5f4ea3ee14c6ef289fd06d502d4e96ba03','sha256:d3eab3bc257a445bd4d31a7849077e037ae6d0d7ce288e35982b24c700e94f2d',NULL,NULL,NULL,NULL,NULL,1789560000000000,1789560000000000,NULL,'{"__record__":"GraphTask","fields":{"agent_id":"agent-golden-revision-1","attempt_count":0,"created_at":{"__datetime__":"2026-09-16T12:00:00Z"},"current_attempt_id":null,"execution_kind":"model","failure_streak":0,"fencing_epoch":0,"job_id":"job-unfinished","latest_checkpoint_id":null,"latest_control_id":null,"latest_result_id":null,"not_before":null,"priority":10,"role":"worker","specification":{"__record__":"GraphTaskSpecification","fields":{"authority":{"__record__":"GraphAuthority","fields":{"access_modes":["read"],"capability_ids":["capability.read"],"connector_ids":[],"contract_bindings":{"capability.read":"sha256:1111111111111111111111111111111111111111111111111111111111111111"},"model_route_ids":[],"operational_effects":["none"],"resource_ids":[],"sensitivity":"restricted","source_ids":[]}},"budgets":[{"__record__":"BudgetAmount","fields":{"amount":3,"dimension":"work_units"}}],"created_by":"owner","description":"Produce the accepted worker result.","expected_result_contract":{"kind":"worker"},"max_steps":2,"max_wall_time_seconds":120,"title":"Worker"}},"state":"ready","superseded_by_task_id":null,"supersedes_task_id":null,"task_id":"worker","task_revision":1,"task_scope_digest":"sha256:d3eab3bc257a445bd4d31a7849077e037ae6d0d7ce288e35982b24c700e94f2d","task_spec_digest":"sha256:e5d8cb59770ee00af6f86aeae6f43f5f4ea3ee14c6ef289fd06d502d4e96ba03","terminal_at":null,"updated_at":{"__datetime__":"2026-09-16T12:00:00Z"}}}');
INSERT INTO "job_tasks" VALUES('agent-golden-revision-1','job-unfinished','finalizer','pending','finalizer','internal_capability',1000,NULL,NULL,1,'sha256:d83364948558e77f689f38826f173661c6f25e4cf2530943b48ed71626fcfdcf','sha256:6fc5d317d040919bebb9dba7e1559a16065f7b5d18425d61cb26cf4505a833b5',NULL,NULL,NULL,NULL,NULL,1789560000000000,1789560000000000,NULL,'{"__record__":"GraphTask","fields":{"agent_id":"agent-golden-revision-1","attempt_count":0,"created_at":{"__datetime__":"2026-09-16T12:00:00Z"},"current_attempt_id":null,"execution_kind":"internal_capability","failure_streak":0,"fencing_epoch":0,"job_id":"job-unfinished","latest_checkpoint_id":null,"latest_control_id":null,"latest_result_id":null,"not_before":null,"priority":1000,"role":"finalizer","specification":{"__record__":"GraphTaskSpecification","fields":{"authority":{"__record__":"GraphAuthority","fields":{"access_modes":[],"capability_ids":[],"connector_ids":[],"contract_bindings":{},"model_route_ids":[],"operational_effects":["none"],"resource_ids":[],"sensitivity":"restricted","source_ids":[]}},"budgets":[{"__record__":"BudgetAmount","fields":{"amount":1,"dimension":"work_units"}}],"created_by":"owner","description":"Promote the authenticated worker result.","expected_result_contract":{"kind":"final"},"max_steps":1,"max_wall_time_seconds":120,"title":"Finalizer"}},"state":"pending","superseded_by_task_id":null,"supersedes_task_id":null,"task_id":"finalizer","task_revision":1,"task_scope_digest":"sha256:6fc5d317d040919bebb9dba7e1559a16065f7b5d18425d61cb26cf4505a833b5","task_spec_digest":"sha256:d83364948558e77f689f38826f173661c6f25e4cf2530943b48ed71626fcfdcf","terminal_at":null,"updated_at":{"__datetime__":"2026-09-16T12:00:00Z"}}}');
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
INSERT INTO "mcp_server_bindings" VALUES('agent-golden-revision-1','mcp-binding-33333333333333333333333333333333','{"__record__":"MCPServerBinding","fields":{"admitted_at":{"__datetime__":"2026-01-02T03:04:05Z"},"authentication_mode":"none","connection_id":null,"endpoint":"https://golden.fixture.test/mcp","keywords":[],"last_checked_at":{"__datetime__":"2026-01-02T03:04:05Z"},"local_label":"Golden MCP","maximum_outbound_sensitivity":"internal","owner_principal_id":"agent-golden-revision-1","protocol_capabilities_digest":null,"protocol_version":"2025-11-25","required_scopes":[],"resource_uri":null,"revision":1,"revoked_at":null,"secret_reference":null,"server_name":null,"server_version":null,"stale_reason":null,"state":"active","summary":"","tools":[{"__record__":"MCPToolBinding","fields":{"access_mode":"read","automation_eligibility":"automation_direct","capability_id":"mcp.tool:sha256:4d3cfc5edea3ece33f6fc5a1d2791d8df939f105c4f984cf69457a3fdd1dff6d","completion_semantics":"direct_result","description":"Read a golden fixture.","executor_id":"mcp.executor:mcp-binding-33333333333333333333333333333333","input_schema":{"additionalProperties":false,"properties":{"term":{"anyOf":[{"type":"string"},{"type":"null"}]}},"required":["term"],"type":"object"},"input_schema_digest":"sha256:8986c3193b313bc3b3a2cc037a884118bc793e1b6cd9d7907b1a85d1789515ad","local_name":"mcp_5dbf912fb554_read","maximum_outbound_sensitivity":"restricted","operational_effect":"none","output_schema":null,"output_schema_digest":null,"presentation_keywords":[],"presentation_load_mode":"on_demand","presentation_summary":"Read a golden fixture.","presentation_text_trust":"admitted_untrusted","presentation_toolbox_id":"sources","presentation_when_to_use":"Read a golden fixture.","raw_input_schema":{"$schema":"http://json-schema.org/draft-07/schema#","additionalProperties":false,"definitions":{"term":{"anyOf":[{"type":"string"},{"type":"null"}]}},"properties":{"term":{"$ref":"#/definitions/term"}},"required":["term"],"type":"object"},"remote_name":"read","result_sensitivity":"internal","task_support":"forbidden"}}],"when_to_use":""}}');
CREATE TABLE messages (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(run_id, position)
);
INSERT INTO "messages" VALUES('run-golden-revision-1',0,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"TextBlock","fields":{"text":"Golden question."}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"user"},"tool_calls":[]}}');
INSERT INTO "messages" VALUES('run-golden-revision-1',1,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"TextBlock","fields":{"text":"Golden answer."}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"assistant"},"tool_calls":[]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',0,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"TextBlock","fields":{"text":"Create two text documents for the registry fixture."}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"user"},"tool_calls":[]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',1,'{"__record__":"CanonicalMessage","fields":{"content":[],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"assistant"},"tool_calls":[{"__record__":"ToolCall","fields":{"arguments":{"tool_names":["artifact_create_document"]},"id":"test-toolbox-load-1","name":"toolbox_load","provider_call_id":null}}]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',2,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"ToolResultBlock","fields":{"call_id":"test-toolbox-load-1","capability_id":null,"executor_id":null,"is_error":false,"output":{"data":{"activation_digest":"sha256:f6739a38e080c02085ff1b2e69091a360fdc8a3cbe38d97d6b912939c7bcf39c","catalog_digest":"sha256:f2fa394032e91d0496f9a5241ce6a978ddcd11d43e6d75d5a39dd0b70415dbbd","contracts":[{"automation_eligibility":"automation_direct","capability_id":"artifact.create_document","complete":false,"connector":null,"contract_digest":"sha256:5a1a0b54dfb064e0d81c13471187de3b605138c2641283fef5f4352f9ff6abaa","effect_evidence_basis":null,"grant_policy":null,"input_schema_digest":"sha256:6ed9551c99a51987e47fd09d15010fd2d9fb1016fea91abd0c3dcaa352794df4","inspection_tool":"toolbox_inspect","origin_revision_digest":"sha256:bd8356a05d9849a634944d029238363636973373d3ae1d3702149d2fdf1b587b","requires_automation_grant":false,"tool_name":"artifact_create_document"}],"definition_bytes":553,"loaded_names":["artifact_create_document"],"run_id":"run-00000000000000000000000000000001"},"kind":"toolbox_load_receipt"},"output_sha256":null,"sensitivity":{"__enum__":"ModelSensitivity","value":"restricted"},"sensitivity_provenance":{"authority":"tool_catalog_control","catalog_digest":"sha256:f2fa394032e91d0496f9a5241ce6a978ddcd11d43e6d75d5a39dd0b70415dbbd","control_name":"toolbox_load","run_id":"run-00000000000000000000000000000001"}}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"tool"},"tool_calls":[]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',3,'{"__record__":"CanonicalMessage","fields":{"content":[],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"assistant"},"tool_calls":[{"__record__":"ToolCall","fields":{"arguments":{"content":"Golden artifact 0.","filename":"golden-0.txt","format":"txt"},"id":"create-0","name":"artifact_create_document","provider_call_id":null}}]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',4,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"ToolResultBlock","fields":{"call_id":"create-0","capability_id":"artifact.create_document","executor_id":"artifact.create_document.executor","is_error":false,"output":{"artifact":{"artifact_id":"artifact-00000000000000000000000000000001","byte_size":18,"call_id":"create-0","capability_id":"artifact.create_document","conversation_id":"conversation-00000000000000000000000000000001","created_at":"2026-10-02T16:54:18.476070Z","filename":"golden-0.txt","media_type":"text/plain","provenance":{"authorship":"model_authored_analysis","columns":[],"derived_from_artifact_id":null,"evidence_call_ids":[],"local_file_binding":null,"parameters_sha256":null,"resource_bindings":[],"result_binding":null,"row_count":null,"sql_fingerprint":null},"run_id":"run-00000000000000000000000000000001","sensitivity":"restricted","sha256":"sha256:641fc61003c7639d83d76af5fe200ee094fc7faeeb48a276131efc79a0e53fee"},"data":{"character_count":18,"format":"txt"},"delivery_status":"not_delivered","kind":"artifact.document"},"output_sha256":"sha256:a9ed8524c6d5a664b08609c66c1b7c8ab2ce5df64e371fe9ae55c581f68899d0","sensitivity":{"__enum__":"ModelSensitivity","value":"restricted"},"sensitivity_provenance":{"authority":"artifact_domain_bound_provenance","authorship":"model_authored_analysis","capability_id":"artifact.create_document","evidence_call_ids":[]}}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"tool"},"tool_calls":[]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',5,'{"__record__":"CanonicalMessage","fields":{"content":[],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"assistant"},"tool_calls":[{"__record__":"ToolCall","fields":{"arguments":{"content":"Golden artifact 1.","filename":"golden-1.txt","format":"txt"},"id":"create-1","name":"artifact_create_document","provider_call_id":null}}]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',6,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"ToolResultBlock","fields":{"call_id":"create-1","capability_id":"artifact.create_document","executor_id":"artifact.create_document.executor","is_error":false,"output":{"artifact":{"artifact_id":"artifact-00000000000000000000000000000002","byte_size":18,"call_id":"create-1","capability_id":"artifact.create_document","conversation_id":"conversation-00000000000000000000000000000001","created_at":"2026-10-02T16:54:18.485295Z","filename":"golden-1.txt","media_type":"text/plain","provenance":{"authorship":"model_authored_analysis","columns":[],"derived_from_artifact_id":null,"evidence_call_ids":[],"local_file_binding":null,"parameters_sha256":null,"resource_bindings":[],"result_binding":null,"row_count":null,"sql_fingerprint":null},"run_id":"run-00000000000000000000000000000001","sensitivity":"restricted","sha256":"sha256:8a5668d1d4e11dc7d2673988d34f99f41e6b58c1b05fa643b0bd31dbc2a0fe01"},"data":{"character_count":18,"format":"txt"},"delivery_status":"not_delivered","kind":"artifact.document"},"output_sha256":"sha256:d367ed73a19c342e68cfb93cb820814b00639d45d252696273c140bd0a527125","sensitivity":{"__enum__":"ModelSensitivity","value":"restricted"},"sensitivity_provenance":{"authority":"artifact_domain_bound_provenance","authorship":"model_authored_analysis","capability_id":"artifact.create_document","evidence_call_ids":[]}}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"tool"},"tool_calls":[]}}');
INSERT INTO "messages" VALUES('run-00000000000000000000000000000001',7,'{"__record__":"CanonicalMessage","fields":{"content":[{"__record__":"TextBlock","fields":{"text":"Golden artifacts created."}}],"provider_id":null,"provider_metadata":{},"role":{"__enum__":"MessageRole","value":"assistant"},"tool_calls":[]}}');
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
INSERT INTO "runs" VALUES('run-golden-revision-1','agent-golden-revision-1','conversation-golden-revision-1',0,'{"__record__":"RunInput","fields":{"agent_id":"agent-golden-revision-1","caller_principal_id":"agent-golden-revision-1","caller_principal_verified":true,"conversation_id":"conversation-golden-revision-1","created_at":{"__datetime__":"2026-01-02T03:04:05Z"},"history_sensitivity":"public","id":"run-golden-revision-1","message":"Golden question.","resolved_source_scope":null,"source_scope_ids":[],"start":{"__record__":"RunStartEnvelope","fields":{"execution_scope":null,"instruction_authority":null,"instruction_digest":null,"origin":"user","payload_digest":null,"trusted_instruction":null,"trusted_instruction_id":null,"untrusted_payload":{},"user_message":"Golden question."}}}}','{"__record__":"LoopExit","fields":{"artifact_deliveries":[],"artifacts":[],"conversation_id":"conversation-golden-revision-1","created_at":{"__datetime__":"2026-01-02T03:04:05Z"},"final_text":"Golden answer.","kind":{"__enum__":"LoopExitKind","value":"completed"},"provider_failure":null,"provider_id":null,"reason":"completed","run_id":"run-golden-revision-1","sensitivity":"restricted","steps":0,"usage":{"__record__":"ModelUsage","fields":{"cache_read_tokens":0,"cache_write_tokens":0,"cost_estimate":{"__record__":"CostEstimate","fields":{"amount_usd":null,"basis":null,"code":"pricing_schedule_unavailable","components":[],"rate_schedule_id":null,"status":{"__enum__":"CostEstimateStatus","value":"unavailable"}}},"input_tokens":0,"output_tokens":0,"reasoning_tokens":0}}}}');
INSERT INTO "runs" VALUES('run-00000000000000000000000000000001','agent-golden-revision-1','conversation-00000000000000000000000000000001',0,'{"__record__":"RunInput","fields":{"agent_id":"agent-golden-revision-1","caller_principal_id":"agent-golden-revision-1","caller_principal_verified":true,"conversation_id":"conversation-00000000000000000000000000000001","created_at":{"__datetime__":"2026-10-02T16:54:18.456771Z"},"history_sensitivity":"public","id":"run-00000000000000000000000000000001","message":"Create two text documents for the registry fixture.","resolved_source_scope":{"resource_ids":[],"source_ids":[]},"source_scope_ids":[],"start":{"__record__":"RunStartEnvelope","fields":{"execution_scope":null,"instruction_authority":null,"instruction_digest":null,"origin":"user","payload_digest":null,"trusted_instruction":null,"trusted_instruction_id":null,"untrusted_payload":{},"user_message":"Create two text documents for the registry fixture."}}}}','{"__record__":"LoopExit","fields":{"artifact_deliveries":[],"artifacts":[{"__record__":"ArtifactRef","fields":{"artifact_id":"artifact-00000000000000000000000000000001","byte_size":18,"call_id":"create-0","capability_id":"artifact.create_document","conversation_id":"conversation-00000000000000000000000000000001","created_at":{"__datetime__":"2026-10-02T16:54:18.476070Z"},"filename":"golden-0.txt","media_type":"text/plain","provenance":{"__record__":"ArtifactProvenance","fields":{"authorship":{"__enum__":"ArtifactAuthorship","value":"model_authored_analysis"},"columns":[],"derived_from_artifact_id":null,"evidence_call_ids":[],"local_file_binding":null,"parameters_sha256":null,"resource_bindings":[],"result_binding":null,"row_count":null,"sql_fingerprint":null}},"run_id":"run-00000000000000000000000000000001","sensitivity":{"__enum__":"Sensitivity","value":"restricted"},"sha256":"sha256:641fc61003c7639d83d76af5fe200ee094fc7faeeb48a276131efc79a0e53fee"}},{"__record__":"ArtifactRef","fields":{"artifact_id":"artifact-00000000000000000000000000000002","byte_size":18,"call_id":"create-1","capability_id":"artifact.create_document","conversation_id":"conversation-00000000000000000000000000000001","created_at":{"__datetime__":"2026-10-02T16:54:18.485295Z"},"filename":"golden-1.txt","media_type":"text/plain","provenance":{"__record__":"ArtifactProvenance","fields":{"authorship":{"__enum__":"ArtifactAuthorship","value":"model_authored_analysis"},"columns":[],"derived_from_artifact_id":null,"evidence_call_ids":[],"local_file_binding":null,"parameters_sha256":null,"resource_bindings":[],"result_binding":null,"row_count":null,"sql_fingerprint":null}},"run_id":"run-00000000000000000000000000000001","sensitivity":{"__enum__":"Sensitivity","value":"restricted"},"sha256":"sha256:8a5668d1d4e11dc7d2673988d34f99f41e6b58c1b05fa643b0bd31dbc2a0fe01"}}],"conversation_id":"conversation-00000000000000000000000000000001","created_at":{"__datetime__":"2026-10-02T16:54:18.491192Z"},"final_text":"Golden artifacts created.","kind":{"__enum__":"LoopExitKind","value":"completed"},"provider_failure":null,"provider_id":null,"reason":"completed","run_id":"run-00000000000000000000000000000001","sensitivity":"restricted","steps":4,"usage":{"__record__":"ModelUsage","fields":{"cache_read_tokens":0,"cache_write_tokens":0,"cost_estimate":{"__record__":"CostEstimate","fields":{"amount_usd":{"__decimal__":"0"},"basis":null,"code":"pricing_schedule_unavailable","components":[{"__record__":"CostComponent","fields":{"amount_usd":{"__decimal__":"0"},"basis":null,"metric":null,"modifiers":[],"name":"attempt","quantity":null,"rate_schedule_id":null,"rate_usd":null,"unit":null,"unit_size":null,"usage_range":null}}],"rate_schedule_id":null,"status":{"__enum__":"CostEstimateStatus","value":"partial"}}},"input_tokens":0,"output_tokens":0,"reasoning_tokens":0}}}}');
CREATE TABLE scheduled_routines (
    agent_id TEXT NOT NULL,
    routine_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    state TEXT NOT NULL,
    next_due_at_us INTEGER,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, routine_id)
);
INSERT INTO "scheduled_routines" VALUES('agent-golden-revision-1','routine-daily','conversation-routine','active',1793525400000000,'{"__record__":"ScheduledRoutine","fields":{"active_occurrence_id":null,"allowed_access_modes":["none"],"allowed_capability_ids":["artifact.create_document"],"allowed_connector_binding_ids":[],"allowed_operational_effects":["none"],"allowed_resource_ids":[],"allowed_source_ids":["source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"],"attempt_count":0,"authorized_instruction":"Summarize yesterday''s orders.","capability_grants":[],"charged_cost_usd":{"__decimal__":"0"},"charged_tokens":0,"consecutive_failures":0,"contract_bindings":{"__record__":"ExecutionContractBindings","fields":{"capability_contracts":{},"model_routes":{}}},"conversation_id":"conversation-routine","created_at":{"__datetime__":"2026-11-01T09:30:00Z"},"cumulative_max_attempts":10,"cumulative_max_cost_usd":null,"cumulative_max_occurrences":10,"cumulative_max_tokens":20000,"distribution_plan":{"__record__":"DistributionPlan","fields":{}},"eligible_model_routes":["mock:released"],"expires_at":{"__datetime__":"2027-01-01T00:00:00Z"},"instruction_digest":"sha256:748284909c4cc0eccafed7c84585fb8880fdeb66897d3d5453095d00a24c62ab","last_acknowledged_precheck_observation":null,"last_delivery_ids":[],"last_occurrence_id":null,"maximum_consecutive_failures":3,"misfire_policy":"latest_only","next_due_at":{"__datetime__":"2026-11-01T09:30:00Z"},"occurrence_count":0,"outcome_contract":{"__record__":"OutcomeContract","fields":{}},"owner_principal_id":"agent:agent-golden-revision-1","per_run_max_cost_usd":null,"per_run_max_tokens":2000,"precheck":null,"promotion_evidence":null,"reporting_mode":"always","reserved_cost_usd":{"__decimal__":"0"},"reserved_tokens":0,"revision":1,"run_immediately":false,"schedule":{"__record__":"IntervalSchedule","fields":{"anchor_at":{"__datetime__":"2026-11-01T09:30:00Z"},"interval_seconds":86400}},"schedule_interpreter_revision":1,"sensitivity_ceiling":"internal","skill_bindings":[],"state":"active","title":"Daily orders","updated_at":{"__datetime__":"2026-11-01T09:30:00Z"}}}');
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
INSERT INTO "snapshots" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','sync-shop','{"__record__": "SourceCatalogSnapshot", "fields": {"facets": [{"__record__": "CatalogFacet", "fields": {"kind": {"__enum__": "FacetKind", "value": "tabular"}, "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "payload": {"columns": {"__packed_columns__": [["id", "TEXT", 0, true], ["CustomerName", "TEXT", 1, true]]}, "indexes": [{"columns": ["CustomerName"], "kind": "btree", "name": "main.customers_index", "predicate": null, "unique": false, "write_conflict_supported": false}], "row_count_estimate": null}, "resource_id": "catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10", "revision": "sha256:28e91c63f50b8172b3769a1d41d417872536991cde6b304f2fa4c956ecb963c5", "sync_id": "sync-shop"}}, {"__record__": "CatalogFacet", "fields": {"kind": {"__enum__": "FacetKind", "value": "tabular"}, "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "payload": {"columns": {"__packed_columns__": [["id", "TEXT", 0, true], ["customer_id", "TEXT", 1, true], ["total", "TEXT", 2, true]]}, "indexes": [], "row_count_estimate": null}, "resource_id": "catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58", "revision": "sha256:7b76761f7cdd0927ec388a324a3a9dc2907994005094ca08466e6d2eb6e24018", "sync_id": "sync-shop"}}, {"__record__": "CatalogFacet", "fields": {"kind": {"__enum__": "FacetKind", "value": "tabular"}, "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "payload": {"columns": {"__packed_columns__": [["id", "TEXT", 0, true], ["order_id", "TEXT", 1, true]]}, "indexes": [], "row_count_estimate": null}, "resource_id": "catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519", "revision": "sha256:af576a60424345c388b0fa819405df9cebc4617edc4a456e1a0634928bbb7929", "sync_id": "sync-shop"}}], "relationships": [{"__record__": "CatalogRelationship", "fields": {"attributes": {}, "confidence": 1.0, "field_pairs": [{"__record__": "RelationshipFieldPair", "fields": {"ordinal": 0, "source_field": "customer_id", "target_field": "id"}}], "from_resource_id": "catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58", "id": "catalog-relationship:sha256:af96691ea213e6a134602e22e45d720d9b33fdf09f4db8e76ba7415be897260e", "kind": {"__enum__": "RelationshipKind", "value": "references"}, "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "provenance": {"__enum__": "RelationshipProvenance", "value": "connector"}, "revision": "sha256:29b23cbc35e26ce33db9cd04b3fccb3b2399fa5132f3c3d70d9a95bf1d9e6a6d", "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67", "sync_id": "sync-shop", "to_resource_id": "catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10"}}, {"__record__": "CatalogRelationship", "fields": {"attributes": {}, "confidence": 1.0, "field_pairs": [{"__record__": "RelationshipFieldPair", "fields": {"ordinal": 0, "source_field": "order_id", "target_field": "id"}}], "from_resource_id": "catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519", "id": "catalog-relationship:sha256:db32d81613d20cc3f1d06a4c6301303f7fb3b8cece33f4f86e98056f16d43f2f", "kind": {"__enum__": "RelationshipKind", "value": "references"}, "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "provenance": {"__enum__": "RelationshipProvenance", "value": "connector"}, "revision": "sha256:30c141c879377d2478c200ed0560ef4313710bb6b33c97678bf8c559f36e8997", "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67", "sync_id": "sync-shop", "to_resource_id": "catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58"}}], "resources": [{"__record__": "CatalogResource", "fields": {"agent_id": "agent-golden-revision-1", "current_revision": "sha256:6fe2b36ca114701509535142fa33cb1fc536d1b4415af5dc3ce8033aaddc6dc4", "current_sync_id": "sync-shop", "external_uri": "sqlite:///main.customers", "first_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "id": "catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10", "kind": {"__enum__": "ResourceKind", "value": "table"}, "last_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "name": "customers", "native_identity": "main.customers", "sensitivity": {"__enum__": "Sensitivity", "value": "unknown"}, "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}}, {"__record__": "CatalogResource", "fields": {"agent_id": "agent-golden-revision-1", "current_revision": "sha256:1510673c8ef18aa53e21276eeddd162931c8f10170721966bee5520d9fdcdad0", "current_sync_id": "sync-shop", "external_uri": "sqlite:///main.orders", "first_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "id": "catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58", "kind": {"__enum__": "ResourceKind", "value": "table"}, "last_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "name": "orders", "native_identity": "main.orders", "sensitivity": {"__enum__": "Sensitivity", "value": "unknown"}, "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}}, {"__record__": "CatalogResource", "fields": {"agent_id": "agent-golden-revision-1", "current_revision": "sha256:e76f501422faa5ba489dea120646aabdc770675bdfc0e3c9f54198d7c9e0ba9a", "current_sync_id": "sync-shop", "external_uri": "sqlite:///main.invoices", "first_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "id": "catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519", "kind": {"__enum__": "ResourceKind", "value": "table"}, "last_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "name": "invoices", "native_identity": "main.invoices", "sensitivity": {"__enum__": "Sensitivity", "value": "unknown"}, "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}}, {"__record__": "CatalogResource", "fields": {"agent_id": "agent-golden-revision-1", "current_revision": "sha256:1adc8785dd971dc96d4f757ab9482b999eee84d9803c2c0a16897613155e06f8", "current_sync_id": "sync-shop", "external_uri": "sqlite:///main", "first_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "id": "catalog-resource:sha256:d5af33d43a44432bf7e3d4387909a628cd37bcff2eb19a5d2b95b02c00d476d7", "kind": {"__enum__": "ResourceKind", "value": "folder"}, "last_observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "name": "main", "native_identity": "main", "sensitivity": {"__enum__": "Sensitivity", "value": "unknown"}, "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}}], "revisions": [{"__record__": "CatalogResourceRevision", "fields": {"facet_revisions": ["sha256:28e91c63f50b8172b3769a1d41d417872536991cde6b304f2fa4c956ecb963c5"], "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "relationship_revisions": ["sha256:29b23cbc35e26ce33db9cd04b3fccb3b2399fa5132f3c3d70d9a95bf1d9e6a6d"], "resource_id": "catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10", "revision": "sha256:6fe2b36ca114701509535142fa33cb1fc536d1b4415af5dc3ce8033aaddc6dc4", "source_revision": null, "sync_id": "sync-shop"}}, {"__record__": "CatalogResourceRevision", "fields": {"facet_revisions": ["sha256:7b76761f7cdd0927ec388a324a3a9dc2907994005094ca08466e6d2eb6e24018"], "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "relationship_revisions": ["sha256:29b23cbc35e26ce33db9cd04b3fccb3b2399fa5132f3c3d70d9a95bf1d9e6a6d", "sha256:30c141c879377d2478c200ed0560ef4313710bb6b33c97678bf8c559f36e8997"], "resource_id": "catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58", "revision": "sha256:1510673c8ef18aa53e21276eeddd162931c8f10170721966bee5520d9fdcdad0", "source_revision": null, "sync_id": "sync-shop"}}, {"__record__": "CatalogResourceRevision", "fields": {"facet_revisions": ["sha256:af576a60424345c388b0fa819405df9cebc4617edc4a456e1a0634928bbb7929"], "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "relationship_revisions": ["sha256:30c141c879377d2478c200ed0560ef4313710bb6b33c97678bf8c559f36e8997"], "resource_id": "catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519", "revision": "sha256:e76f501422faa5ba489dea120646aabdc770675bdfc0e3c9f54198d7c9e0ba9a", "source_revision": null, "sync_id": "sync-shop"}}, {"__record__": "CatalogResourceRevision", "fields": {"facet_revisions": [], "observed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "relationship_revisions": [], "resource_id": "catalog-resource:sha256:d5af33d43a44432bf7e3d4387909a628cd37bcff2eb19a5d2b95b02c00d476d7", "revision": "sha256:1adc8785dd971dc96d4f757ab9482b999eee84d9803c2c0a16897613155e06f8", "source_revision": null, "sync_id": "sync-shop"}}], "sync": {"__record__": "CatalogSync", "fields": {"adapter_id": "sqlite", "agent_id": "agent-golden-revision-1", "completed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "error_code": null, "id": "sync-shop", "relationship_count": 2, "resource_count": 4, "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67", "source_revision": null, "started_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "status": {"__enum__": "CatalogSyncStatus", "value": "succeeded"}}}}}');
CREATE TABLE source_read_scopes (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, source_id),
    FOREIGN KEY (agent_id, source_id)
        REFERENCES sources(agent_id, id)
        ON DELETE CASCADE
);
INSERT INTO "source_read_scopes" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','{"__record__":"SourceReadScope","fields":{"mode":"all","resource_ids":[]}}');
CREATE TABLE sources (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
INSERT INTO "sources" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','{"__record__":"SourceRegistration","fields":{"adapter_id":"postgresql","agent_id":"agent-golden-revision-1","attached_at":{"__datetime__":"2026-11-01T09:30:00Z"},"configuration":{},"detached_at":null,"display_name":"Shop","id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67","keywords":[],"native_identity":"shop","presentation_sensitivity":"restricted","summary":"","when_to_use":""}}');
CREATE TABLE syncs (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
INSERT INTO "syncs" VALUES('agent-golden-revision-1','sync-shop','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','{"__record__": "CatalogSync", "fields": {"adapter_id": "sqlite", "agent_id": "agent-golden-revision-1", "completed_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "error_code": null, "id": "sync-shop", "relationship_count": 2, "resource_count": 4, "source_id": "source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67", "source_revision": null, "started_at": {"__datetime__": "2026-11-01T09:30:00Z"}, "status": {"__enum__": "CatalogSyncStatus", "value": "succeeded"}}}');
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
CREATE INDEX artifacts_by_conversation ON artifacts
    (agent_id, state, conversation_id, created_at_us, artifact_id);
CREATE INDEX artifacts_by_run ON artifacts (agent_id, run_id, state);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('job_graph_events',1);
COMMIT;
