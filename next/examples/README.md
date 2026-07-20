# Daita 2.0 retained examples

These examples use only the isolated v2 package in `next/src`. They never read
v1 state and never use the implicit user-home state root. Unless `--root` is
provided, each walkthrough creates a fresh temporary v2 root and removes it on
exit. A supplied root should be dedicated to that example.

The data walkthroughs use a tiny scripted provider so they are deterministic,
offline, and free of credentials. Production providers are configured through
`daita.create_llm_provider()` and secret providers or provider SDK environment
configuration; no example contains an API key.

Run a walkthrough from `next/`:

```bash
python examples/00_quickstart_sqlite_from_db.py
python examples/01_inspectable_operation.py
python examples/02_catalog_assisted_joins.py
python examples/03_governed_reads_and_writes.py
python examples/04_persistent_runtime_store.py
python examples/06_memory_for_business_semantics.py
python examples/07_monitor_orders.py
python examples/09_custom_data_plugin_extension.py
python examples/10_csv_to_sqlite_data_app.py
```

`03_governed_reads_and_writes.py` deliberately stops at a pending approval. It
does not approve or execute the write. A human must inspect the durable impact
evidence and make an explicit approval or rejection decision.

The former `05_data_quality_and_lineage.py` and
`08_infrastructure_catalog.py` examples are deferred from the 2.0 MVP. Their
v1 plugin-specific behavior needs dedicated v2 capability/resource adapters;
placeholder implementations would create false compatibility, so those files
are intentionally absent.

The production-shaped foreground host example lives in
`deployments/data-team-agent/`. Its root is required, provider credentials are
resolved outside the example, and writable source access is never enabled.
