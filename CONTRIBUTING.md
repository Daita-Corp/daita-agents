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

1. Subclass `BaseLLMProvider` from `daita.llm.base`
2. Implement `_generate_impl()` and `_stream_impl()`
3. Add it to `daita/llm/factory.py`
4. Add tests in `tests/unit/`

### Adding a New Plugin

1. Create a new file in `daita/plugins/`
2. Subclass `BasePlugin` from `daita.plugins.base`
3. Implement `get_tools()` returning a list of `AgentTool`
4. Export it from `daita/plugins/__init__.py`

## Code Style

- Formatting is enforced by `black` via the pre-commit hook — just commit and it runs automatically
- Type hints on all public functions
- Docstrings on all public classes and methods

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
