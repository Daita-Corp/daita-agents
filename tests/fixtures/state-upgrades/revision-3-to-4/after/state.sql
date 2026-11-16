BEGIN TRANSACTION;
CREATE TABLE agent_home_migrations (
    revision INTEGER NOT NULL PRIMARY KEY,
    migration_id TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL
);
INSERT INTO "agent_home_migrations" VALUES(1,'agent_home_revision_1','a08bdc56e3cb7c3dbe77dc0d8b8ed9aac1299a6a701902dba8e11a5ebe70f25e');
INSERT INTO "agent_home_migrations" VALUES(2,'0002_durable_adaptive_task_graph','cb98e978fcb6db71cc1cb3ee5532d92ce99117db7359a008a98a4651df1c20e2');
INSERT INTO "agent_home_migrations" VALUES(3,'0003_framework_caller_authority','3357ee7ad8153e5160d06133082be89dca1e868f1529029f5b63988fae5dca42');
INSERT INTO "agent_home_migrations" VALUES(4,'0004_foreground_analysis_evidence','462d5ad981152f9c7fb5f8d7253a7e692b87ffc33eb84ae67c079e8091a6957b');
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
CREATE TABLE catalog_changes (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    sync_id TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('added', 'changed', 'removed')),
    resource_id TEXT NOT NULL,
    native_identity TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, source_id, sequence)
);
CREATE TABLE catalog_links (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE TABLE catalog_relationships (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    id TEXT NOT NULL,
    from_resource_id TEXT NOT NULL,
    to_resource_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, id)
);
INSERT INTO "catalog_relationships" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-relationship:sha256:af96691ea213e6a134602e22e45d720d9b33fdf09f4db8e76ba7415be897260e','catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58','catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10','sha256:29b23cbc35e26ce33db9cd04b3fccb3b2399fa5132f3c3d70d9a95bf1d9e6a6d','{"__record__":"CatalogRelationship","fields":{"attributes":{},"confidence":1.0,"field_pairs":[{"__record__":"RelationshipFieldPair","fields":{"ordinal":0,"source_field":"customer_id","target_field":"id"}}],"from_resource_id":"catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58","id":"catalog-relationship:sha256:af96691ea213e6a134602e22e45d720d9b33fdf09f4db8e76ba7415be897260e","kind":{"__enum__":"RelationshipKind","value":"references"},"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"provenance":{"__enum__":"RelationshipProvenance","value":"connector"},"revision":"sha256:29b23cbc35e26ce33db9cd04b3fccb3b2399fa5132f3c3d70d9a95bf1d9e6a6d","source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67","sync_id":"sync-shop","to_resource_id":"catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10"}}');
INSERT INTO "catalog_relationships" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-relationship:sha256:db32d81613d20cc3f1d06a4c6301303f7fb3b8cece33f4f86e98056f16d43f2f','catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519','catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58','sha256:30c141c879377d2478c200ed0560ef4313710bb6b33c97678bf8c559f36e8997','{"__record__":"CatalogRelationship","fields":{"attributes":{},"confidence":1.0,"field_pairs":[{"__record__":"RelationshipFieldPair","fields":{"ordinal":0,"source_field":"order_id","target_field":"id"}}],"from_resource_id":"catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519","id":"catalog-relationship:sha256:db32d81613d20cc3f1d06a4c6301303f7fb3b8cece33f4f86e98056f16d43f2f","kind":{"__enum__":"RelationshipKind","value":"references"},"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"provenance":{"__enum__":"RelationshipProvenance","value":"connector"},"revision":"sha256:30c141c879377d2478c200ed0560ef4313710bb6b33c97678bf8c559f36e8997","source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67","sync_id":"sync-shop","to_resource_id":"catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58"}}');
CREATE TABLE catalog_resources (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    id TEXT NOT NULL,
    kind TEXT NOT NULL,
    native_identity TEXT NOT NULL,
    name TEXT NOT NULL,
    revision TEXT NOT NULL,
    source_object_id TEXT,
    detail TEXT NOT NULL CHECK (detail IN ('inventory', 'complete')),
    last_seen_sync TEXT NOT NULL,
    first_observed_at TEXT NOT NULL,
    last_observed_at TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, id)
);
INSERT INTO "catalog_resources" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10','table','main.customers','customers','sha256:6fe2b36ca114701509535142fa33cb1fc536d1b4415af5dc3ce8033aaddc6dc4',NULL,'complete','sync-shop','2026-11-01T09:30:00Z','2026-11-01T09:30:00Z','{"__record__":"CatalogEntry","fields":{"detailed":true,"facets":[{"__record__":"CatalogFacet","fields":{"kind":{"__enum__":"FacetKind","value":"tabular"},"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"payload":{"columns":{"__packed_columns__":[["id","TEXT",0,true],["CustomerName","TEXT",1,true]]},"indexes":[{"columns":["CustomerName"],"kind":"btree","name":"main.customers_index","predicate":null,"unique":false,"write_conflict_supported":false}],"row_count_estimate":null},"resource_id":"catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10","revision":"sha256:28e91c63f50b8172b3769a1d41d417872536991cde6b304f2fa4c956ecb963c5","sync_id":"sync-shop"}}],"resource":{"__record__":"CatalogResource","fields":{"agent_id":"agent-golden-revision-1","current_revision":"sha256:6fe2b36ca114701509535142fa33cb1fc536d1b4415af5dc3ce8033aaddc6dc4","current_sync_id":"sync-shop","external_uri":"sqlite:///main.customers","first_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"id":"catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10","kind":{"__enum__":"ResourceKind","value":"table"},"last_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"name":"customers","native_identity":"main.customers","sensitivity":{"__enum__":"Sensitivity","value":"unknown"},"source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}},"revision":{"__record__":"CatalogResourceRevision","fields":{"facet_revisions":["sha256:28e91c63f50b8172b3769a1d41d417872536991cde6b304f2fa4c956ecb963c5"],"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"relationship_revisions":["sha256:29b23cbc35e26ce33db9cd04b3fccb3b2399fa5132f3c3d70d9a95bf1d9e6a6d"],"resource_id":"catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10","revision":"sha256:6fe2b36ca114701509535142fa33cb1fc536d1b4415af5dc3ce8033aaddc6dc4","source_revision":null,"sync_id":"sync-shop"}},"source_object_id":null}}');
INSERT INTO "catalog_resources" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58','table','main.orders','orders','sha256:1510673c8ef18aa53e21276eeddd162931c8f10170721966bee5520d9fdcdad0',NULL,'complete','sync-shop','2026-11-01T09:30:00Z','2026-11-01T09:30:00Z','{"__record__":"CatalogEntry","fields":{"detailed":true,"facets":[{"__record__":"CatalogFacet","fields":{"kind":{"__enum__":"FacetKind","value":"tabular"},"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"payload":{"columns":{"__packed_columns__":[["id","TEXT",0,true],["customer_id","TEXT",1,true],["total","TEXT",2,true]]},"indexes":[],"row_count_estimate":null},"resource_id":"catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58","revision":"sha256:7b76761f7cdd0927ec388a324a3a9dc2907994005094ca08466e6d2eb6e24018","sync_id":"sync-shop"}}],"resource":{"__record__":"CatalogResource","fields":{"agent_id":"agent-golden-revision-1","current_revision":"sha256:1510673c8ef18aa53e21276eeddd162931c8f10170721966bee5520d9fdcdad0","current_sync_id":"sync-shop","external_uri":"sqlite:///main.orders","first_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"id":"catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58","kind":{"__enum__":"ResourceKind","value":"table"},"last_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"name":"orders","native_identity":"main.orders","sensitivity":{"__enum__":"Sensitivity","value":"unknown"},"source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}},"revision":{"__record__":"CatalogResourceRevision","fields":{"facet_revisions":["sha256:7b76761f7cdd0927ec388a324a3a9dc2907994005094ca08466e6d2eb6e24018"],"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"relationship_revisions":["sha256:29b23cbc35e26ce33db9cd04b3fccb3b2399fa5132f3c3d70d9a95bf1d9e6a6d","sha256:30c141c879377d2478c200ed0560ef4313710bb6b33c97678bf8c559f36e8997"],"resource_id":"catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58","revision":"sha256:1510673c8ef18aa53e21276eeddd162931c8f10170721966bee5520d9fdcdad0","source_revision":null,"sync_id":"sync-shop"}},"source_object_id":null}}');
INSERT INTO "catalog_resources" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519','table','main.invoices','invoices','sha256:e76f501422faa5ba489dea120646aabdc770675bdfc0e3c9f54198d7c9e0ba9a',NULL,'complete','sync-shop','2026-11-01T09:30:00Z','2026-11-01T09:30:00Z','{"__record__":"CatalogEntry","fields":{"detailed":true,"facets":[{"__record__":"CatalogFacet","fields":{"kind":{"__enum__":"FacetKind","value":"tabular"},"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"payload":{"columns":{"__packed_columns__":[["id","TEXT",0,true],["order_id","TEXT",1,true]]},"indexes":[],"row_count_estimate":null},"resource_id":"catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519","revision":"sha256:af576a60424345c388b0fa819405df9cebc4617edc4a456e1a0634928bbb7929","sync_id":"sync-shop"}}],"resource":{"__record__":"CatalogResource","fields":{"agent_id":"agent-golden-revision-1","current_revision":"sha256:e76f501422faa5ba489dea120646aabdc770675bdfc0e3c9f54198d7c9e0ba9a","current_sync_id":"sync-shop","external_uri":"sqlite:///main.invoices","first_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"id":"catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519","kind":{"__enum__":"ResourceKind","value":"table"},"last_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"name":"invoices","native_identity":"main.invoices","sensitivity":{"__enum__":"Sensitivity","value":"unknown"},"source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}},"revision":{"__record__":"CatalogResourceRevision","fields":{"facet_revisions":["sha256:af576a60424345c388b0fa819405df9cebc4617edc4a456e1a0634928bbb7929"],"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"relationship_revisions":["sha256:30c141c879377d2478c200ed0560ef4313710bb6b33c97678bf8c559f36e8997"],"resource_id":"catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519","revision":"sha256:e76f501422faa5ba489dea120646aabdc770675bdfc0e3c9f54198d7c9e0ba9a","source_revision":null,"sync_id":"sync-shop"}},"source_object_id":null}}');
INSERT INTO "catalog_resources" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:d5af33d43a44432bf7e3d4387909a628cd37bcff2eb19a5d2b95b02c00d476d7','folder','main','main','sha256:1adc8785dd971dc96d4f757ab9482b999eee84d9803c2c0a16897613155e06f8',NULL,'complete','sync-shop','2026-11-01T09:30:00Z','2026-11-01T09:30:00Z','{"__record__":"CatalogEntry","fields":{"detailed":true,"facets":[],"resource":{"__record__":"CatalogResource","fields":{"agent_id":"agent-golden-revision-1","current_revision":"sha256:1adc8785dd971dc96d4f757ab9482b999eee84d9803c2c0a16897613155e06f8","current_sync_id":"sync-shop","external_uri":"sqlite:///main","first_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"id":"catalog-resource:sha256:d5af33d43a44432bf7e3d4387909a628cd37bcff2eb19a5d2b95b02c00d476d7","kind":{"__enum__":"ResourceKind","value":"folder"},"last_observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"name":"main","native_identity":"main","sensitivity":{"__enum__":"Sensitivity","value":"unknown"},"source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}},"revision":{"__record__":"CatalogResourceRevision","fields":{"facet_revisions":[],"observed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"relationship_revisions":[],"resource_id":"catalog-resource:sha256:d5af33d43a44432bf7e3d4387909a628cd37bcff2eb19a5d2b95b02c00d476d7","revision":"sha256:1adc8785dd971dc96d4f757ab9482b999eee84d9803c2c0a16897613155e06f8","source_revision":null,"sync_id":"sync-shop"}},"source_object_id":null}}');
CREATE TABLE catalog_search (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    name_key TEXT NOT NULL,
    identity_key TEXT NOT NULL,
    words TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, resource_id)
);
INSERT INTO "catalog_search" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:ea6544d1e25ee1d511b0cdebb8bc763e41b94bdf54ddac3ecf4dabff73142e10','customers','main.customers',' customer customers id main name table tables ','[["column","CustomerName"],["column","id"],["index_field","CustomerName"],["kind","table tables"],["native_identity","main.customers"],["relationship_field","id"],["resource_name","customers"]]');
INSERT INTO "catalog_search" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:96ab60367dae5b655ac0fff862772f95fd6db3ddd84a4f8fa19b8706e350fd58','orders','main.orders',' customer id main orders table tables total ','[["column","customer_id"],["column","id"],["column","total"],["kind","table tables"],["native_identity","main.orders"],["relationship_field","customer_id"],["relationship_field","id"],["resource_name","orders"]]');
INSERT INTO "catalog_search" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:b88c3eafd2efa118a033158179135a3aeaf2260271bfe95b0bc4aa3cff30a519','invoices','main.invoices',' id invoices main order table tables ','[["column","id"],["column","order_id"],["kind","table tables"],["native_identity","main.invoices"],["relationship_field","order_id"],["resource_name","invoices"]]');
INSERT INTO "catalog_search" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','catalog-resource:sha256:d5af33d43a44432bf7e3d4387909a628cd37bcff2eb19a5d2b95b02c00d476d7','main','main',' folder folders main ','[["kind","folder folders"],["native_identity","main"],["resource_name","main"]]');
CREATE TABLE catalog_source_state (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, source_id)
);
INSERT INTO "catalog_source_state" VALUES('agent-golden-revision-1','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','{"__record__":"CatalogSourceState","fields":{"agent_id":"agent-golden-revision-1","current_sync_id":"sync-shop","keep_current":true,"last_change_at":null,"last_error_code":null,"last_full_check_at":null,"last_missed":null,"last_probe_at":null,"probe_cursor":null,"probe_tokens":{},"source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"}}');
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
INSERT INTO "jobs" VALUES('agent-golden-revision-1','job-unfinished','conversation-1','failed',1,1789560000000000,1789560000000000,'{"__record__":"Job","fields":{"agent_id":"agent-golden-revision-1","answers":[],"checklist":[],"conversation_id":"conversation-1","created_at":{"__datetime__":"2026-09-16T12:00:00Z"},"current_run_id":null,"delegations":[],"failure_code":"job_upgraded","final_summary":"Created by an older version of Daita; start it again.","job_id":"job-unfinished","objective":"Complete the deterministic graph test.","question":null,"revision":1,"run_count":0,"run_limit":20,"sensitivity":"restricted","source_ids":[],"state":"failed","token_limit":1000000,"tokens_used":0,"updated_at":{"__datetime__":"2026-09-16T12:00:00Z"}}}');
CREATE TABLE mcp_server_bindings (
    agent_id TEXT NOT NULL,
    binding_id TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, binding_id)
);
INSERT INTO "mcp_server_bindings" VALUES('agent-golden-revision-1','mcp-binding-33333333333333333333333333333333','{"__record__":"MCPServerBinding","fields":{"admitted_at":{"__datetime__":"2026-01-02T03:04:05Z"},"authentication_mode":"none","connection_id":null,"endpoint":"https://golden.fixture.test/mcp","keywords":[],"last_checked_at":{"__datetime__":"2026-01-02T03:04:05Z"},"local_label":"Golden MCP","maximum_outbound_sensitivity":"internal","owner_principal_id":"agent-golden-revision-1","protocol_capabilities_digest":null,"protocol_version":"2025-11-25","required_scopes":[],"resource_uri":null,"revision":1,"revoked_at":null,"secret_reference":null,"server_name":null,"server_version":null,"stale_reason":null,"state":"active","summary":"","tools":[{"__record__":"MCPToolBinding","fields":{"access_mode":"read","capability_id":"mcp.tool:sha256:4d3cfc5edea3ece33f6fc5a1d2791d8df939f105c4f984cf69457a3fdd1dff6d","completion_semantics":"direct_result","description":"Read a golden fixture.","executor_id":"mcp.executor:mcp-binding-33333333333333333333333333333333","input_schema":{"additionalProperties":false,"properties":{"term":{"anyOf":[{"type":"string"},{"type":"null"}]}},"required":["term"],"type":"object"},"input_schema_digest":"sha256:8986c3193b313bc3b3a2cc037a884118bc793e1b6cd9d7907b1a85d1789515ad","local_name":"mcp_5dbf912fb554_read","maximum_outbound_sensitivity":"restricted","operational_effect":"none","output_schema":null,"output_schema_digest":null,"presentation_keywords":[],"presentation_load_mode":"on_demand","presentation_summary":"Read a golden fixture.","presentation_text_trust":"admitted_untrusted","presentation_toolbox_id":"sources","presentation_when_to_use":"Read a golden fixture.","raw_input_schema":{"$schema":"http://json-schema.org/draft-07/schema#","additionalProperties":false,"definitions":{"term":{"anyOf":[{"type":"string"},{"type":"null"}]}},"properties":{"term":{"$ref":"#/definitions/term"}},"required":["term"],"type":"object"},"remote_name":"read","result_sensitivity":"internal","task_support":"forbidden"}}],"when_to_use":""}}');
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
INSERT INTO "scheduled_routines" VALUES('agent-golden-revision-1','routine-daily','conversation-routine','active',1793525400000000,'{"__record__":"ScheduledRoutine","fields":{"active_occurrence_id":null,"attempt_count":0,"authorized_instruction":"Summarize yesterday''s orders.","charged_cost_usd":{"__decimal__":"0"},"charged_tokens":0,"consecutive_failures":0,"conversation_id":"conversation-routine","created_at":{"__datetime__":"2026-11-01T09:30:00Z"},"instruction_digest":"sha256:748284909c4cc0eccafed7c84585fb8880fdeb66897d3d5453095d00a24c62ab","last_delivery_ids":[],"last_occurrence_id":null,"next_due_at":{"__datetime__":"2026-11-01T09:30:00Z"},"occurrence_count":0,"owner_principal_id":"agent:agent-golden-revision-1","revision":1,"schedule":{"__record__":"IntervalSchedule","fields":{"anchor_at":{"__datetime__":"2026-11-01T09:30:00Z"},"interval_seconds":86400}},"schedule_interpreter_revision":1,"source_ids":["source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67"],"state":"active","title":"Daily orders","updated_at":{"__datetime__":"2026-11-01T09:30:00Z"}}}');
CREATE TABLE semantic_annotations (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
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
INSERT INTO "syncs" VALUES('agent-golden-revision-1','sync-shop','source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67','{"__record__":"CatalogSync","fields":{"adapter_id":"sqlite","agent_id":"agent-golden-revision-1","completed_at":{"__datetime__":"2026-11-01T09:30:00Z"},"error_code":null,"id":"sync-shop","relationship_count":2,"resource_count":4,"source_id":"source:sha256:3b1347ae31c8d9053145b2eb505ddc3466a8cc7ab65cc3ee700581cf29a36a67","source_revision":null,"started_at":{"__datetime__":"2026-11-01T09:30:00Z"},"status":{"__enum__":"CatalogSyncStatus","value":"succeeded"},"truncated":false}}');
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
CREATE INDEX catalog_resources_by_source
    ON catalog_resources(agent_id, source_id, native_identity);
CREATE INDEX catalog_relationships_by_source
    ON catalog_relationships(agent_id, source_id);
CREATE INDEX catalog_relationships_from
    ON catalog_relationships(agent_id, from_resource_id);
CREATE INDEX catalog_relationships_to
    ON catalog_relationships(agent_id, to_resource_id);
CREATE INDEX catalog_search_by_source ON catalog_search(agent_id, source_id);
CREATE INDEX catalog_search_by_name ON catalog_search(agent_id, name_key);
CREATE INDEX catalog_search_by_identity ON catalog_search(agent_id, identity_key);
COMMIT;
