# Durable-job live benchmarks

This directory separates paid model variability from deterministic failure and
load mechanics. All paid modules are skipped unless their exact authorization
variable is set. Every live `Agent.run` retains the ordinary 24-step,
100,000-token, 300-second functional safety envelope and a per-interaction
estimated-cost ceiling. Token use is recorded as benchmark evidence rather than
enforced through a tighter run limit that can stop the behavior under evaluation.
The default cost ceiling is `$0.15`; override it only with an explicitly
authorized positive value in `DAITA_STAGE_B_BENCHMARK_MAX_COST_USD`.

Supply one model with `DAITA_STAGE_B_BENCHMARK_MODEL_IDS` or a comma-separated
release-reviewed model matrix. Supply credentials through the generic
`DAITA_STAGE_B_BENCHMARK_LLM_API_KEY` or a provider-specific variable such as
`DAITA_STAGE_B_BENCHMARK_OPENAI_API_KEY`.

| Module | Purpose | Explicit authorization | Maximum live interactions |
| --- | --- | --- | --- |
| `test_paraphrases.py` | Natural user phrasing for reads, starts, global status, result recovery, and cancellation | `DAITA_RUN_LIVE_STAGE_B_PARAPHRASE_BENCHMARK=1` | 16 for the default model |
| `test_catalog_scale.py` | Exact resource selection among 16, 64, and 128 look-alike tables | `DAITA_RUN_LIVE_STAGE_B_CATALOG_BENCHMARK=1` | 6 for the default model |
| `test_model_matrix.py` | Per-model immediate, cross-conversation result, start, and cancel certification | `DAITA_RUN_LIVE_STAGE_B_MODEL_MATRIX=1` | 4 per configured model |

Run collection without paid calls:

```bash
pytest tests/live/benchmarks --collect-only -o addopts="--tb=short -q --strict-markers"
```

Deterministic provider-failure and benchmark-support qualification live in the
default owner suite:

```bash
pytest tests/jobs/test_provider_failure_after_admission.py tests/jobs/test_benchmark_harness.py -v
```

Run the deterministic soak explicitly:

```bash
DAITA_RUN_STAGE_B_CONCURRENCY_SOAK=1 \
pytest tests/slow/test_job_concurrency.py -o addopts="--tb=short -q --strict-markers" -v
```

For a live module, export its exact authorization variable and benchmark API
key, then invoke only that module. Do not combine all live modules unless the
sum of their interaction and cost ceilings has been explicitly authorized.
