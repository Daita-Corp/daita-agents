# Graph-effect fixtures

`native-upsert.json` is the declarative catalog, permission, and input fixture
for the opt-in Phase 8 live-model PostgreSQL graph test. The test routes adapter
I/O to the repository's transactional fake database; it does not need database
credentials or contact PostgreSQL.

The MCP graph test uses `httpx.MockTransport` and the existing fake server in
`tests/support/mcp_actions.py`. Both tests therefore spend only the explicitly
authorized model budget while retaining exact local evidence about dispatches,
commits, and receipts.

Run one test explicitly after providing a reviewed provider key:

```shell
DAITA_RUN_LIVE_GRAPH_EFFECTS=1 \
DAITA_GRAPH_EFFECTS_LIVE_LLM_API_KEY=... \
pytest tests/live/jobs/test_graph_effects.py -v -s
```

The default per-model-run estimated-cost ceiling is USD 0.15. A graph task may
make up to three attempts, so the conservative ceiling for both tests together
is USD 1.20. Select one test with `-k` to reduce that ceiling by half.
