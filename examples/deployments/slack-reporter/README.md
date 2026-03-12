# slack-reporter

A single agent that runs on a schedule, queries a PostgreSQL database for sales
metrics, and posts a formatted digest to a Slack channel.

**Use case:** Daily sales digest posted to `#analytics` every morning at 9 AM.

## Highlights

- PostgreSQL plugin for database queries
- Slack plugin for posting messages
- Cron scheduling via `daita-project.yaml` — no extra code needed
- Demonstrates composing multiple plugins in one agent

## Required environment variables

```
OPENAI_API_KEY      sk-...
DATABASE_URL        postgresql://user:pass@host:5432/dbname
SLACK_BOT_TOKEN     xoxb-...
SLACK_CHANNEL_ID    #analytics   (or a channel ID like C0123456789)
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_CHANNEL_ID=#analytics
```

### Slack app permissions

Your Slack bot needs the following OAuth scopes:
- `chat:write` — post messages to channels
- `channels:read` — verify channel access

## Run manually

```bash
# Trigger the digest immediately (for testing)
python run.py
```

## Schedule

The cron schedule in `daita-project.yaml` runs the agent every day at 09:00 UTC:

```yaml
schedules:
  - name: daily_digest
    agent: reporter_agent
    cron: "0 9 * * *"
    prompt: "Run the daily sales digest for today and post it to Slack."
    timezone: "UTC"
```

Change the cron expression or timezone to match your team's schedule.

## Test

```bash
# Fast tests — no services required
pytest tests/ -v

# Full integration — requires all env vars
pytest tests/ -v
```

## Expected database schema

The agent assumes these tables (adjust the prompt in `agents/reporter_agent.py` for your schema):

```sql
orders      (id, customer_id, status, total_amount, created_at)
order_items (id, order_id, product_id, quantity, unit_price)
products    (id, name, category)
customers   (id, name, email)
```

## Deploy

```bash
daita push
```

After deploying, the agent will run automatically according to the cron schedule.
