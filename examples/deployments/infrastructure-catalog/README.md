# infrastructure-catalog

A single agent that discovers data stores across AWS accounts and GitHub
repositories, builds a unified catalog, and answers questions about the
organization's data landscape.

**Use case:** "What production databases do we have?" -> the agent scans AWS
RDS/DynamoDB/S3/ElastiCache/Redshift/API Gateway across regions, scans GitHub repos for
connection strings, deduplicates results, and returns a categorized summary.

## Highlights

- CatalogPlugin with pluggable discoverers and profilers
- AWSDiscoverer — scans RDS, DynamoDB, S3, ElastiCache, Redshift, API Gateway across regions
- GitHubScanner — finds connection strings in `.env`, `docker-compose.yml`, `database.yml`, etc.
- Store deduplication — same database found by AWS and GitHub is merged into one record
- Environment inference — tags stores as production/staging/dev from naming patterns
- Schema profiling — drill into any discovered store to extract full table/column metadata
- Paginated tool responses — handles large catalogs without blowing the LLM context

## Required environment variables

```
OPENAI_API_KEY        sk-...
```

Plus AWS credentials via one of:
```
AWS_ACCESS_KEY_ID     + AWS_SECRET_ACCESS_KEY
AWS_PROFILE           named profile from ~/.aws/config
AWS_ROLE_ARN          cross-account role to assume
```

## Optional environment variables

```
AWS_REGIONS           comma-separated (default: us-east-1)
AWS_SERVICES          comma-separated (default: rds,dynamodb,s3,elasticache,redshift,apigateway)
GITHUB_TOKEN          GitHub personal access token
GITHUB_ORG            scan all repos in this org
GITHUB_REPOS          comma-separated owner/repo list
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export AWS_PROFILE=my-profile        # or set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
export GITHUB_TOKEN=ghp_...          # optional
export GITHUB_ORG=my-org             # optional
```

## Run

```bash
# Demo mode — discover, summarize, and answer sample questions
python run.py

# One-shot question
python run.py "What databases do we have in us-west-2?"

# Custom regions
AWS_REGIONS=us-east-1,us-west-2,eu-west-1 python run.py
```

## Test

```bash
# Fast tests — no cloud credentials required
pytest tests/ -v
```

## Project structure

```
infrastructure-catalog/
├── agents/
│   └── catalog_agent.py    # Agent definition + discoverer configuration
├── workflows/              # No workflows — single-agent example
├── data/                   # Not used in this example
├── tests/
│   └── test_basic.py       # Unit tests (no credentials needed)
├── daita-project.yaml      # Project config
├── requirements.txt
└── run.py                  # Entry point
```

## Deploy

```bash
daita push
```
