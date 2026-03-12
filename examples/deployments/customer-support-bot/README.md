# customer-support-bot

A two-agent workflow that reads incoming support tickets, classifies them
by category, and routes them to the appropriate specialist agent.

**Pipeline:**
```
Classifier → billing_queue   → Billing Specialist
Classifier → technical_queue → Technical Specialist
Classifier → general_queue   → General Specialist
```

**Use case:** Support queue automation that tags and triages tickets before
a human ever sees them.

## Highlights

- Conditional relay routing (unlike the linear pattern in `etl-pipeline`)
- `MemoryPlugin` for per-conversation history across the classifier and specialists
- Agent specialization — each specialist has a narrow, well-defined prompt
- Pure-Python tools (`classify_ticket`, `draft_response`) — fully testable without LLM

## Required environment variables

```
OPENAI_API_KEY   sk-...
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Run

```bash
# Demo mode — processes 3 sample tickets
python run.py

# Process a single ticket
python run.py "Cannot log in" "I get a 403 error when I try to access my account."
```

## Test

```bash
# Fast tests — no API key needed (classify_ticket and draft_response are pure Python)
pytest tests/ -v

# LLM integration — requires OPENAI_API_KEY
pytest tests/ -v
```

## How routing works

The Classifier outputs one of three routing values in its JSON response:
- `billing_queue` → Billing Specialist
- `technical_queue` → Technical Specialist
- `general_queue` → General Specialist

The `workflow.connect()` calls wire each channel to the right specialist.
If the Classifier outputs `technical_queue`, only the Technical Specialist runs.

## Project structure

```
customer-support-bot/
├── agents/
│   ├── classifier.py        # Classifies tickets and routes to queues
│   └── specialist.py        # Billing, Technical, and General specialists
├── workflows/
│   └── support_workflow.py  # Routing workflow
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
