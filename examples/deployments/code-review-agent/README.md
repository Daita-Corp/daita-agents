# Code Review Agent

A single agent that reviews Python code for security vulnerabilities and quality
issues using the **skills system** — composable units of capability that bundle
tools with domain-specific instructions.

## Skills used

| Skill | Pattern | What it does |
|-------|---------|-------------|
| `SecurityReviewSkill` | `BaseSkill` subclass + `instructions_file` | Regex-based vulnerability scanning (SQL injection, command injection, hardcoded secrets, insecure deserialization, path traversal, SSRF) + input validation analysis |
| `code_quality` | `Skill()` factory + inline instructions | AST-based cyclomatic complexity analysis + PEP 8 naming convention checks |

The agent also has a standalone `read_file` tool for loading source files.

## Why skills?

Without skills, you'd pass tools + a massive prompt to the agent. Skills let you:
- **Package domain knowledge** alongside the tools that use it
- **Compose capabilities** — add/remove skills without touching the agent prompt
- **Reuse across agents** — the same `SecurityReviewSkill` works in any agent

## Required env vars

```
OPENAI_API_KEY
```

## Running

```bash
# Review the included sample file (has intentional issues)
python run.py

# Review your own file
python run.py path/to/your_code.py

# Review with a specific focus
python run.py path/to/file.py "focus on authentication"
```

## Testing

```bash
# Unit tests (no API key needed)
pytest tests/

# Including integration test (needs OPENAI_API_KEY)
OPENAI_API_KEY=sk-... pytest tests/ -v
```
