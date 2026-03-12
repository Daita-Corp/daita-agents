# deep-research

A four-agent pipeline that produces a structured research report with citations
for any query, using live web search.

**Pipeline:**
```
Orchestrator → (research_plan) → Web Researcher → (raw_findings) → Analyst → (synthesis) → Report Writer
```

**Use case:** "What are the latest breakthroughs in solid-state batteries?" →
receive a structured markdown report with key findings, analysis, and citations.

## Highlights

- Full relay pipeline across four specialised agents
- `WebSearchPlugin` (Tavily) for live web search
- `MemoryPlugin` for shared research context
- `LineagePlugin` on the Analyst for data flow tracking
- Pure-Python tools fully tested without any API keys

## Required environment variables

```
OPENAI_API_KEY    sk-...
TAVILY_API_KEY    tvly-...   (free tier: 1000 searches/month at tavily.com)
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

## Run

```bash
# Demo query
python run.py

# Custom query
python run.py "What are the economic impacts of large language models?"
```

## Test

```bash
# Fast tests — no API keys required (all tool functions are pure Python)
pytest tests/ -v

# Full integration — requires OPENAI_API_KEY and TAVILY_API_KEY
pytest tests/ -v
```

## How it works

| Agent | Input | Output | Tools / Plugins |
|---|---|---|---|
| Orchestrator | Research query | Research plan (JSON) | `decompose_query`, MemoryPlugin |
| Web Researcher | Research plan | Raw findings (JSON) | WebSearchPlugin, MemoryPlugin |
| Analyst | Raw findings | Synthesis (JSON) | `extract_scope`, MemoryPlugin, LineagePlugin |
| Report Writer | Synthesis | Markdown report | `format_citation`, `build_report_structure`, MemoryPlugin |

## Project structure

```
deep-research/
├── agents/
│   ├── orchestrator.py     # Plans the research strategy
│   ├── researcher.py       # Searches the web for each sub-question
│   ├── analyst.py          # Synthesises findings into insights
│   └── report_writer.py    # Writes the final cited report
├── workflows/
│   └── research_workflow.py  # Four-agent pipeline
├── tests/
│   └── test_basic.py
├── daita-project.yaml
├── requirements.txt
└── run.py
```

## Deploy

```bash
daita push
```
