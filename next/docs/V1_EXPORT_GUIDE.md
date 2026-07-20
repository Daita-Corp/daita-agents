# V1 export guidance

The supported candidate path is a reviewed fresh start, not an internal-state
copy. Before retiring v1, operators should preserve only information that has
an independent business or audit reason to survive.

## Inventory before export

- Agent names and intended owners.
- User-authored instructions and configuration, excluding credentials.
- Attached source identities and least-privilege access requirements.
- Session transcripts that policy permits retaining.
- Explicit user corrections, reviewed memories, and accepted skill text.
- Monitor definitions, schedules, and delivery expectations.
- Approval and evidence records required by an external audit policy.

Export through v1's documented public inspection or database-specific backup
mechanism. Do not treat private Python objects, pickles, caches, provider-native
payloads, lock files, or v1 SQLite tables as a v2 import format.

## Recreate in v2

Initialize a fresh agent, reattach each source, and re-enter only reviewed
configuration through v2 public APIs. Persist a secret reference such as an
environment-variable or keychain identifier; never paste the resolved value
into a source configuration, transcript, memory, skill, or migration note.

Compare catalog identities, representative read results, memories, skills,
and monitors against the export. Any unsupported v1 feature must follow the
[support matrix](SUPPORT_MATRIX.md); it must not be recovered by importing or
executing v1 runtime code.
