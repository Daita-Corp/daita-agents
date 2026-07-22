# Contributing to Daita Agents

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Getting Started

### Prerequisites

- Python 3.11+
- `pip` or `uv`

### Setup

```bash
git clone https://github.com/daita-tech/daita-agents.git
cd daita-agents
pip install -e ".[dev]"
pre-commit install
```

The last step installs the git hook that automatically formats your code with `black` before every commit.

### Running Tests

```bash
pytest tests/ -m "not requires_llm and not requires_db"
```

## How to Contribute

### Reporting Bugs

Open a [GitHub Issue](https://github.com/daita-tech/daita-agents/issues) with:
- A clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

### Suggesting Features

Open a [GitHub Issue](https://github.com/daita-tech/daita-agents/issues) with the `enhancement` label. Describe the use case and why it would benefit the community.

### Submitting a Pull Request

1. Fork the repo and create a branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes and add tests where applicable.
3. Ensure tests pass:
   ```bash
   pytest tests/ -m "not requires_llm and not requires_db"
   ```
4. Open a PR against `main` with a clear description of what changed and why.

### Adding a New LLM Provider

1. Implement the `ModelProvider` contract under `src/daita/llm/providers/`.
2. Keep provider-native payloads and translation inside that adapter.
3. Lazy-import the optional SDK and add it to the matching extra.
4. Register construction in `src/daita/llm/factory.py`.
5. Add focused provider translation and routing tests under `tests/`.

### Adding a Source or Data Capability

Extend the existing adapter, catalog, capability registry, and data-domain
owners. Keep structural discovery in the catalog path, source I/O behind the
adapter boundary, and concrete validation immediately before execution. Do not
add a plugin hierarchy, alternate catalog, or second agent loop. See
`AGENTS.md` for the complete architecture and test requirements.

## Code Style

- Formatting is enforced by `black` via the pre-commit hook — just commit and it runs automatically
- Type hints on all public functions
- Docstrings on all public classes and methods

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
