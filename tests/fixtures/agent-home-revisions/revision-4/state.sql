BEGIN TRANSACTION;
CREATE TABLE agent_home_migrations (
    revision INTEGER NOT NULL PRIMARY KEY,
    migration_id TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL
);
INSERT INTO "agent_home_migrations" VALUES(1,'agent_home_revision_1','a08bdc56e3cb7c3dbe77dc0d8b8ed9aac1299a6a701902dba8e11a5ebe70f25e');
INSERT INTO "agent_home_migrations" VALUES(2,'0002_durable_adaptive_task_graph','cb98e978fcb6db71cc1cb3ee5532d92ce99117db7359a008a98a4651df1c20e2');
INSERT INTO "agent_home_migrations" VALUES(3,'0003_framework_caller_authority','3357ee7ad8153e5160d06133082be89dca1e868f1529029f5b63988fae5dca42');
INSERT INTO "agent_home_migrations" VALUES(4,'0004_foreground_analysis_evidence','a1b159370b43f1c9859f477e95cab29d1b7ea3bc1c91ee7945a4e98429a3535b');
CREATE TABLE analysis_evidence (
    agent_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('generation', 'cell', 'child', 'run')),
    data TEXT NOT NULL,
    reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes >= 0),
    PRIMARY KEY (run_id, evidence_id)
);
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
INSERT INTO "artifacts" VALUES('artifact-00000000000000000000000000000001','agent-golden-revision-1','run-00000000000000000000000000000001','conversation-00000000000000000000000000000001','agent-golden-revision-1','ready',18,1790960058476070,'{"__record__":"ArtifactRecord","fields":{"agent_id":"agent-golden-revision-1","caller_principal_id":"agent-golden-revision-1","computation_evidence":{},"ref":{"__record__":"ArtifactRef","fields":{"artifact_id":"artifact-00000000000000000000000000000001","byte_size":18,"call_id":"create-0","capability_id":"artifact.create_document","conversation_id":"conversation-00000000000000000000000000000001","created_at":{"__datetime__":"2026-10-02T16:54:18.476070Z"},"filename":"golden-0.txt","media_type":"text/plain","provenance":{"__record__":"ArtifactProvenance","fields":{"authorship":{"__enum__":"ArtifactAuthorship","value":"model_authored_analysis"},"columns":[],"derived_from_artifact_id":null,"evidence_call_ids":[],"local_file_binding":null,"parameters_sha256":null,"resource_bindings":[],"result_binding":null,"row_count":null,"sql_fingerprint":null}},"run_id":"run-00000000000000000000000000000001","sensitivity":{"__enum__":"Sensitivity","value":"restricted"},"sha256":"sha256:641fc61003c7639d83d76af5fe200ee094fc7faeeb48a276131efc79a0e53fee"}},"state":{"__enum__":"ArtifactState","value":"ready"}}}');
CREATE TABLE catalog_links (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
INSERT INTO "catalog_links" VALUES('agent-golden-revision-1','catalog-link:sha256:335e62baace965f276aae9a82ff9c8ffbdde2734afb86e1e713dd30a22888a1a','{"__record__":"CatalogLink","fields":{"agent_id":"agent-golden-revision-1","declared_at":{"__datetime__":"2026-01-02T03:04:05Z"},"declared_revisions":["sha256:1111111111111111111111111111111111111111111111111111111111111111","sha256:2222222222222222222222222222222222222222222222222222222222222222"],"field_pairs":[{"__record__":"RelationshipFieldPair","fields":{"ordinal":0,"source_field":"customer_id","target_field":"id"}}],"from_resource_id":"catalog-resource:sha256:9d6ad160aca152ad7831ea2f5077fc4e85e2aa99c7bdc7f1ec19706ccaf9bf39","from_source_id":"source-golden-orders","id":"catalog-link:sha256:335e62baace965f276aae9a82ff9c8ffbdde2734afb86e1e713dd30a22888a1a","note":"Golden declared link.","to_resource_id":"catalog-resource:sha256:5c75cf1bc66f85b76a03f57a7e20e1288c4c0b2708304a0a157a1e7f80fdbca2","to_source_id":"source-golden-customers"}}');
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
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id),
    UNIQUE(agent_id, run_id, call_id),
    UNIQUE(agent_id, operation_key),
    FOREIGN KEY(agent_id, job_id) REFERENCES jobs(agent_id, job_id)
        ON DELETE RESTRICT
);
CREATE TABLE jobs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'running','waiting_for_you','done','failed','cancelled'
    )),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id)
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
CREATE INDEX analysis_evidence_by_agent_run ON analysis_evidence (agent_id, run_id);
CREATE INDEX jobs_by_state ON jobs (agent_id, state, updated_at_us);
CREATE INDEX effect_receipts_unresolved
    ON effect_receipts(agent_id, unresolved, routine_id, run_id);
CREATE INDEX effect_receipts_grant_reservations
    ON effect_receipts(agent_id, occurrence_id, grant_digest);
CREATE INDEX effect_receipts_by_job ON effect_receipts(agent_id, job_id, unresolved);
COMMIT;
