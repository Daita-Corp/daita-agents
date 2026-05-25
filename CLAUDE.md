# AGENTS.md

Guide for AI assistants (and humans) working on this codebase.

## What this project is

`daita-agents` is a data-focused Python framework for building production-ready AI agents. It ships with a minimal core (`pip install daita-agents`) and a rich set of optional extras (databases, vector stores, cloud services, additional LLM providers). A separate `daita-client` package handles programmatic agent execution.

## Directory layout

```
daita/
  agents/       # Agent and BaseAgent — the main user-facing classes
  llm/          # LLM providers (OpenAI, Anthropic, Gemini, Grok, Mock)
  plugins/      # Infrastructure integrations (databases, APIs, storage)
  core/         # Internals: tools, workflow, relay, streaming, tracing, exceptions
  config/       # AgentConfig, RetryPolicy, RetryStrategy
  cli/          # `daita` CLI commands
  display/      # Console output utilities
  execution.py  # Thin shim re-exporting DaitaClient from daita-client package

tests/
  unit/         # Fast, isolated tests — always runnable
  mocks/        # MockLLMProvider and external service mocks
  fixtures/     # Sample data and test servers
```

## Development setup

```bash
pip install -e ".[dev]"
pre-commit install        # installs black auto-format hook
```

## Running tests

```bash
# Safe default — no API keys or databases required
pytest tests/ -m "not requires_llm and not requires_db"

# Full suite (requires credentials)
pytest tests/

# Single file
pytest tests/unit/test_agent_tools.py -v
```

`asyncio_mode = "auto"` is set in `pyproject.toml` — do not add `@pytest.mark.asyncio` to individual tests.

Available markers: `requires_llm`, `requires_db`, `slow`, `stress`, `unit`, `integration`, `performance`.

## Refactoring discipline

Before introducing any new helper, abstraction, module, base class, builder, registry, or shared utility:

1. Check whether an existing module already owns that responsibility.
2. Prefer extending the existing owner over creating a parallel abstraction.
3. If the change only reduces repetition, do not add a new abstraction unless at least 3 current call sites need it and the abstraction removes more complexity than it adds.
4. For broad standardization, implement one representative file first, run tests, then pause and explain whether the pattern should become project-wide before touching additional files.
5. Avoid churn-only consistency changes. Refactor where the existing code is actively large, complex, buggy, or hard to test.
6. Never create a private framework inside one package when a broader framework layer already exists.

Before editing during a refactor, briefly identify:

- Existing owner of this behavior
- Why the current code is painful
- Smallest change that fixes it
- Why the change is not adding a parallel abstraction
- Tests that will catch behavior drift

### Root-cause fixes

When fixing bugs or reliability issues, trace the failure to the underlying contract, state ownership, lifecycle, or architectural boundary that allowed it. Do not stack narrow patches, special cases, retries, or defensive checks on top of a broken design indefinitely. Prefer replacing the incorrect mechanism with a coherent owner and removing the obsolete path it supersedes, with regression tests that prove the root issue stays fixed.

### Catalog ownership

The catalog plugin is the owner for cataloging infrastructure, normalized schemas, relationships, and graph traversal/search over data assets. `Agent.from_db()` should use the catalog as its source of structural truth when planning queries, finding tables, resolving joins, and traversing relationships. Do not move catalog graphing, infrastructure discovery, or relationship-search ownership into the from_db runtime; from_db should consume catalog capabilities to make querying easier.

## Critical: the lazy import rule

**All optional dependencies must be imported inside `connect()` or inside a `@property client` body — never at module top-level.**

This is the single most important rule. Violating it breaks installs for users who don't have that optional package installed.

**LLM provider pattern** (`@property client`):

```python
# daita/llm/myprovider.py
class MyProvider(BaseLLMProvider):
    def __init__(self, ...):
        self._client = None   # do NOT import the SDK here

    @property
    def client(self):
        if self._client is None:
            try:
                import mypkg                          # lazy — only when first used
                self._client = mypkg.AsyncClient(...)
            except ImportError:
                raise ImportError(
                    "mypkg is required. Install with: pip install 'daita-agents[myprovider]'"
                )
        return self._client
```

**Plugin pattern** (`connect()`):

```python
# daita/plugins/myplugin.py
class MyPlugin(BasePlugin):
    async def connect(self):
        try:
            import mypkg                              # lazy — only when connecting
        except ImportError:
            raise ImportError(
                "mypkg is required. Install with: pip install 'daita-agents[myplugin]'"
            )
        self._client = mypkg.Client(...)
```

Rules:

- Always raise `ImportError`, never `RuntimeError`
- Install hint must use the `pip install 'daita-agents[extra]'` format, not `pip install <pkg>`
- Use `if TYPE_CHECKING:` for type-hint-only imports to avoid circular imports

## Optional dependency → extra mapping

When a plugin or provider needs a package, add it to the matching extra in `pyproject.toml`, not to `[dependencies]`.

| Extra           | Packages                                                                               |
| --------------- | -------------------------------------------------------------------------------------- |
| `anthropic`     | anthropic                                                                              |
| `google`        | google-genai                                                                           |
| `postgresql`    | asyncpg, psycopg2-binary                                                               |
| `mysql`         | aiomysql, SQLAlchemy                                                                   |
| `mongodb`       | motor                                                                                  |
| `sqlite`        | aiosqlite, SQLAlchemy                                                                  |
| `snowflake`     | snowflake-connector-python                                                             |
| `elasticsearch` | elasticsearch                                                                          |
| `chromadb`      | chromadb                                                                               |
| `pinecone`      | pinecone                                                                               |
| `qdrant`        | qdrant-client                                                                          |
| `aws`           | boto3                                                                                  |
| `slack`         | slack-sdk                                                                              |
| `mcp`           | mcp                                                                                    |
| `websearch`     | tavily-python                                                                          |
| `neo4j`         | neo4j                                                                                  |
| `redis`         | redis                                                                                  |
| `data`          | pandas, numpy, openpyxl, beautifulsoup4, lxml, jsonpath-ng, thefuzz, psycopg2, asyncpg |

## Adding a new LLM provider

1. Create `daita/llm/myprovider.py`, subclass `BaseLLMProvider`
2. Implement `_generate_impl()` and `_stream_impl()` (see `daita/llm/base.py` for contracts)
3. Lazy-import the SDK inside `@property client`
4. Register it in `daita/llm/factory.py`
5. Add the package to a new or existing extra in `pyproject.toml`
6. Add tests in `tests/unit/`

## Adding a new plugin

1. Create `daita/plugins/myplugin.py`, subclass `BasePlugin` (or `BaseDatabasePlugin` for DB plugins)
2. Implement `connect()` with lazy import, and `get_tools()` returning `List[AgentTool]`
3. Export from `daita/plugins/__init__.py`
4. Add the package to a new or existing extra in `pyproject.toml`
5. Export from `daita/__init__.py` if it should be part of the top-level public API

## Things to avoid

- **No top-level optional imports.** See the lazy import rule above.
- **No new core dependencies.** If a package is only needed by one plugin or provider, it belongs in an optional extra, not `[dependencies]`.
- **Don't skip pre-commit / black.** Formatting is enforced automatically on commit.
- **Don't add `@pytest.mark.asyncio`.** It's set globally via `asyncio_mode = "auto"`.

## Key files

| File                       | Purpose                                               |
| -------------------------- | ----------------------------------------------------- |
| `daita/__init__.py`        | Public API surface — what users import                |
| `daita/agents/agent.py`    | Main `Agent` class                                    |
| `daita/agents/base.py`     | `BaseAgent` — extend this for custom agents           |
| `daita/llm/base.py`        | `BaseLLMProvider` — extend this for new LLM providers |
| `daita/plugins/base.py`    | `BasePlugin` — extend this for new plugins            |
| `daita/plugins/base_db.py` | `BaseDatabasePlugin` — extend this for DB plugins     |
| `daita/core/tools.py`      | `tool` decorator, `AgentTool`, `ToolRegistry`         |
| `daita/core/exceptions.py` | All exception types                                   |
| `daita/llm/factory.py`     | `create_llm_provider()` — provider registry           |
| `daita/execution.py`       | Re-exports `DaitaClient` from `daita-client` package  |

## More

See [CONTRIBUTING.md](CONTRIBUTING.md) for PR workflow and code style guidelines.
