# Production-shaped local data-team agent

This example runs one explicit foreground `AgentHost` behind its private Unix
socket. The operator must provide a dedicated v2 `--root`; the process never
falls back to `~/.daita-next`. Use `--create` exactly once, then omit it to open
the durable identity on later starts.

```bash
python examples/deployments/data-team-agent/run.py \
  --root /srv/daita/data-team \
  --agent data-team \
  --model openai:gpt-4.1-mini \
  --create
```

Provider SDKs resolve credentials outside this repository (for example from a
process environment or host secret facility). Do not put keys in arguments,
source code, the state root, or this deployment directory.

An optional `--sqlite /absolute/path/sales.db` attachment is read-only. This
deployment intentionally has no automatic write approval path. Send mutable
requests through the local socket with stable idempotency keys and make each
approval or rejection as a separate operator decision.

Use `--dry-run` to validate and print the non-secret deployment shape without
creating state, loading a provider SDK, or opening a socket.
