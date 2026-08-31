# Durable-job live benchmarks

This directory separates paid model variability from deterministic failure and
load mechanics. All paid modules are skipped unless their exact authorization
variable is set. Every live `Agent.run` retains a 12-step, 30,000-token,
180-second, and per-interaction estimated-cost ceiling. The default ceiling is
`$0.15`; override it only with an explicitly authorized positive value in
`DAITA_STAGE_B_BENCHMARK_MAX_COST_USD`.

Supply one model with `DAITA_STAGE_B_BENCHMARK_MODEL_IDS` or a comma-separated
release-reviewed model matrix. Supply credentials through the generic
`DAITA_STAGE_B_BENCHMARK_LLM_API_KEY` or a provider-specific variable such as
`DAITA_STAGE_B_BENCHMARK_OPENAI_API_KEY`.

| Module | Purpose | Explicit authorization | Maximum live interactions |
| --- | --- | --- | --- |
| `test_stage_b_paraphrases_live.py` | Natural user phrasing for reads, starts, global status, result recovery, and cancellation | `DAITA_RUN_LIVE_STAGE_B_PARAPHRASE_BENCHMARK=1` | 16 for the default model |
| `test_stage_b_catalog_scale_live.py` | Exact resource selection among 16, 64, and 128 look-alike tables | `DAITA_RUN_LIVE_STAGE_B_CATALOG_BENCHMARK=1` | 6 for the default model |
| `test_stage_b_model_matrix_live.py` | Per-model immediate, cross-conversation result, start, and cancel certification | `DAITA_RUN_LIVE_STAGE_B_MODEL_MATRIX=1` | 4 per configured model |
| `test_stage_b_provider_failures.py` | Job independence after deterministic provider failures | none; no paid calls | 0 |
| `test_stage_b_concurrency_soak.py` | Concurrent admission, claims, source limits, cancellation, fencing, and reopen | `DAITA_RUN_STAGE_B_CONCURRENCY_SOAK=1` | 0 |

Run collection without paid calls:

```bash
pytest tests/live/benchmarks --collect-only
```

Run the deterministic failure contracts:

```bash
pytest tests/live/benchmarks/test_stage_b_provider_failures.py -v
```

Run the deterministic soak explicitly:

```bash
DAITA_RUN_STAGE_B_CONCURRENCY_SOAK=1 \
pytest tests/live/benchmarks/test_stage_b_concurrency_soak.py -v
```

For a live module, export its exact authorization variable and benchmark API
key, then invoke only that module. Do not combine all live modules unless the
sum of their interaction and cost ceilings has been explicitly authorized.
