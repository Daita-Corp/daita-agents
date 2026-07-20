# Phase 9 candidate baselines

These focused tests are regression tripwires for the local replacement
candidate, not capacity claims or production SLAs. Each workload uses bounded,
deterministic inputs and records its observed measurements as JUnit properties
when the suite is run with `--junitxml`.

| Area | Workload | Ceiling or invariant | Why this boundary exists |
| --- | --- | --- | --- |
| Loop | 8 persisted read operations, 2 model calls each | batch <= 8 s; p95 <= 2 s | Wide enough for shared CI while detecting blocking I/O or accidental extra loop work. |
| Model usage | Same loop workload | exactly 2 calls, 30 input and 10 output tokens per operation | A deterministic semantic budget is stronger than a timing-only signal. |
| Observation | One 512-byte padded payload per operation | canonical payload <= 2 KiB | Detects accidental projection of requests, transcripts, or unbounded evidence. |
| SQLite/WAL | Incremental state for the 8-operation loop workload | <= 16 MiB | Allows page/WAL variability while catching runaway checkpoint duplication. |
| Blob store | 32 logical writes of the same 4 KiB content | exactly 1 physical object; all files <= 128 KiB | Proves content-addressed deduplication and bounds manifest overhead. |
| Catalog | One atomic snapshot of 1,000 resources and 40 exact searches | commit <= 8 s; search p95 <= 1 s | A compact local-catalog scale check with intentionally conservative CI timing. |
| Contention | 32 distinct catalog commits over 4 SQLite connections | 100% commit; batch <= 10 s | Exercises real SQLite writer contention and its busy-timeout/serialization path. |
| Monitors | 12 durable run-now occurrences plus exact replays | 100% success; p95 <= 2 s; no replay model calls | Reliability and exactly-once replay matter more than raw scheduling throughput. |

Timing failures should be reproduced before tightening or relaxing a ceiling.
Correctness invariants must not be weakened to accommodate a slower machine.
