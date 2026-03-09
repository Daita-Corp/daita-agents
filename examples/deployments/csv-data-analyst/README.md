# csv-data-analyst

Ask natural language questions about any CSV file. The agent inspects the
file schema, picks the right pandas operation, and returns a clear answer.

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# Run demo questions against the included sample dataset
python run.py

# Analyse your own file
python run.py data/my_data.csv

# One-shot question
python run.py data/my_data.csv "What are the top 5 products by revenue?"
```

## What it can answer

- **Rankings** — "Top 10 customers by spend", "Lowest performing regions"
- **Aggregations** — "Total revenue by category", "Average order size per month"
- **Distributions** — "How many orders per product?", "What's the breakdown by region?"
- **Filtered views** — "Show Electronics orders over $5000", "Orders where units > 20"
- **Descriptive stats** — "Summarise the revenue column", "Min and max order size"

## Sample dataset

`data/sample_sales.csv` — 96 rows of fictional 2024 sales data.

| Column | Type | Example |
|---|---|---|
| date | string | 2024-01-05 |
| product | string | Laptop |
| category | string | Electronics |
| region | string | North |
| units_sold | int | 3 |
| unit_price | float | 1200.00 |
| revenue | float | 3600.00 |
| customer_id | string | CUST-042 |

## Project structure

```
csv-data-analyst/
├── agents/
│   └── csv_analyst.py    # Agent + 6 pandas tools
├── data/
│   └── sample_sales.csv  # Included sample dataset
├── tests/
│   └── test_analyst.py   # Tool unit tests + LLM integration tests
├── run.py                # Entry point
├── daita-project.yaml    # Project config
└── requirements.txt
```

## Tools

| Tool | What it does |
|---|---|
| `load_csv` | Inspect schema, dtypes, row count, preview |
| `get_summary_stats` | Descriptive statistics for numeric columns |
| `aggregate` | Group by + sum / mean / count / min / max |
| `top_n` | Sort and return top or bottom N rows |
| `count_values` | Frequency table for a categorical column |
| `filter_and_summarise` | Filter with a query expression + optional summary |

## Running tests

```bash
# Tool unit tests — no API key needed
pytest tests/ -k "not Integration"

# Full suite including LLM calls
pytest tests/
```

## Deploy

```bash
export DAITA_API_KEY=your-key-here
daita push
```
