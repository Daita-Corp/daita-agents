# Contributing to Daita Agents

[`AGENTS.md`](AGENTS.md) defines the current architecture, trust boundaries,
dependency policy, and change discipline. Read it before changing production
code.

## Setup

Daita supports Python 3.11 and 3.12. Create a dedicated environment from the
repository root:

```bash
git clone https://github.com/Daita-Corp/daita-agents.git
cd daita-agents
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Report a problem

Open a [GitHub issue](https://github.com/Daita-Corp/daita-agents/issues) with:

- a concise description;
- steps to reproduce;
- expected and observed behavior;
- the Daita and Python versions;
- the operating system; and
- a minimal traceback or log with credentials and sensitive data removed.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

For a feature request, describe the user problem and why the existing
capabilities do not address it. Avoid proposing a parallel loop, runtime,
catalog, scheduler, state store, or writer.

## Submit a change

1. Fork the repository and branch from `main`.
2. Make the smallest complete change at the component responsible for the
   contract.
3. Add focused tests for behavior changes.
4. Run the relevant focused tests and the deterministic suite.
5. Open a pull request that explains the problem, the chosen boundary, and the
   validation performed.

Use type hints on public functions and docstrings on public classes and
methods. Format Python with Black.

```bash
.venv/bin/python -m pytest tests/ -m "not requires_llm and not requires_db"
.venv/bin/python -m black --check src tests
.venv/bin/python -m mypy src/daita tests
```

Live model and external database tests require explicit authorization and
credentials. Do not use paid tests to diagnose a deterministic failure.

## Add a model provider

1. Implement `ModelProvider` under `src/daita/llm/providers/`.
2. Keep provider-native payloads and translation inside the adapter.
3. Import the provider SDK lazily at first use and add its bounded version to
   the default production dependencies in `pyproject.toml`.
4. Register construction in `src/daita/llm/factory.py`.
5. Add focused provider translation, error, and routing tests.

Provider SDKs are part of the complete installation; `dev` is the only
optional dependency group.

## Add a source or data capability

Extend the existing adapter, catalog, registry, and concrete capability domain.
Keep structural discovery in the catalog, source I/O behind the adapter, and
current validation immediately before execution. Return bounded,
schema-validated output through `CapabilityRuntime` and add focused contract
coverage plus one public end-to-end test.

## License

Contributions are licensed under the [MIT License](LICENSE).
