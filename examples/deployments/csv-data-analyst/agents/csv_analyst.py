"""
CSV Analyst Agent

Answers natural language questions about CSV files using pandas.
The agent inspects the file schema first, then calls the appropriate
analysis tools to answer the user's question.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from daita import Agent
from daita.core.tools import tool


@tool
async def load_csv(filepath: str) -> Dict[str, Any]:
    """
    Load a CSV file and return its schema, shape, and a preview of the data.

    Always call this first before answering any question so you understand
    the column names, data types, and what the data looks like.

    Args:
        filepath: Path to the CSV file

    Returns:
        Schema info including columns, dtypes, row count, and first 5 rows
    """
    try:
        df = pd.read_csv(filepath)
        return {
            "filepath": filepath,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "preview": df.head(5).to_dict(orient="records"),
            "null_counts": df.isnull().sum().to_dict(),
        }
    except FileNotFoundError:
        return {"error": f"File not found: {filepath}. Check the path and try again."}
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_summary_stats(
    filepath: str, columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get descriptive statistics (count, mean, min, max, std, quartiles) for numeric columns.

    Args:
        filepath: Path to the CSV file
        columns: Specific columns to summarise. If omitted, all numeric columns are included.

    Returns:
        Descriptive statistics for each selected numeric column
    """
    try:
        df = pd.read_csv(filepath)
        if columns:
            df = df[columns]
        stats = df.describe().round(2).to_dict()
        return {"statistics": stats}
    except Exception as e:
        return {"error": str(e)}


@tool
async def aggregate(
    filepath: str,
    group_by: str,
    agg_column: str,
    agg_function: str,
) -> Dict[str, Any]:
    """
    Group the data by one column and aggregate another.

    Args:
        filepath: Path to the CSV file
        group_by: Column to group by (e.g. "region", "category")
        agg_column: Column to aggregate (e.g. "revenue", "units_sold")
        agg_function: Aggregation to apply — one of: sum, mean, count, min, max

    Returns:
        Aggregated results sorted descending by value
    """
    valid_functions = {"sum", "mean", "count", "min", "max"}
    if agg_function not in valid_functions:
        return {
            "error": f"agg_function must be one of: {', '.join(sorted(valid_functions))}"
        }

    try:
        df = pd.read_csv(filepath)
        result = (
            df.groupby(group_by)[agg_column]
            .agg(agg_function)
            .round(2)
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={agg_column: f"{agg_function}_{agg_column}"})
        )
        return {
            "group_by": group_by,
            "agg_column": agg_column,
            "agg_function": agg_function,
            "results": result.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def top_n(
    filepath: str,
    sort_column: str,
    n: int = 10,
    ascending: bool = False,
    columns_to_show: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return the top (or bottom) N rows sorted by a column.

    Args:
        filepath: Path to the CSV file
        sort_column: Column to sort by (e.g. "revenue")
        n: Number of rows to return (default 10)
        ascending: False for top N (highest first), True for bottom N (lowest first)
        columns_to_show: Columns to include in output. If omitted, all columns are shown.

    Returns:
        Top N rows as a list of records
    """
    try:
        df = pd.read_csv(filepath)
        df_sorted = df.sort_values(sort_column, ascending=ascending).head(n)
        if columns_to_show:
            df_sorted = df_sorted[columns_to_show]
        return {
            "sort_column": sort_column,
            "n": n,
            "ascending": ascending,
            "results": df_sorted.round(2).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def count_values(filepath: str, column: str, top_n: int = 20) -> Dict[str, Any]:
    """
    Count occurrences of each unique value in a column (frequency table).

    Useful for questions like "how many orders per region?" or
    "which products appear most often?"

    Args:
        filepath: Path to the CSV file
        column: Column to count values in
        top_n: Maximum number of values to return (default 20)

    Returns:
        Value counts sorted by frequency descending
    """
    try:
        df = pd.read_csv(filepath)
        counts = df[column].value_counts().head(top_n).reset_index()
        counts.columns = [column, "count"]
        return {
            "column": column,
            "total_unique_values": df[column].nunique(),
            "results": counts.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def filter_and_summarise(
    filepath: str,
    query: str,
    summarise_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Filter rows using a pandas query expression, then optionally summarise a column.

    Query syntax uses column names directly. String values must be quoted.
    Examples:
        "region == 'North'"
        "revenue > 5000 and category == 'Electronics'"
        "units_sold >= 10 and units_sold <= 50"

    Args:
        filepath: Path to the CSV file
        query: Pandas query string to filter rows
        summarise_column: If provided, return sum/mean/count for this column in the filtered set

    Returns:
        Matching rows (up to 50) and optional summary stats
    """
    try:
        df = pd.read_csv(filepath)
        filtered = df.query(query)

        result: Dict[str, Any] = {
            "query": query,
            "matching_rows": len(filtered),
            "total_rows": len(df),
            "results": filtered.head(50).round(2).to_dict(orient="records"),
        }

        if summarise_column and summarise_column in filtered.columns:
            col = filtered[summarise_column]
            result["summary"] = {
                "column": summarise_column,
                "sum": round(float(col.sum()), 2),
                "mean": round(float(col.mean()), 2),
                "min": round(float(col.min()), 2),
                "max": round(float(col.max()), 2),
            }

        return result
    except Exception as e:
        return {"error": str(e)}


def create_agent(csv_path: Optional[str] = None) -> Agent:
    default_path = csv_path or os.getenv("CSV_FILE", "data/sample_sales.csv")

    return Agent(
        name="CSV Analyst",
        model="gpt-4o-mini",
        prompt=f"""You are a data analyst who answers questions about CSV files using pandas tools.

Default file: {default_path}

Your process for every question:
1. LOAD: Call load_csv to inspect the schema — column names, types, and a row preview.
   Skip this step only if you have already loaded the file in this session.
2. ANALYSE: Choose the right tool for the question:
   - Totals / averages by group  → aggregate
   - Highest or lowest records   → top_n
   - Distribution of a category  → count_values
   - Filtered subset             → filter_and_summarise
   - Overall numeric summary     → get_summary_stats
3. ANSWER: Present the results clearly. Include the actual numbers from the tool output.
   Format tables as markdown when there are multiple rows.

Rules:
- Always use the actual column names returned by load_csv — never guess them.
- If a query fails, report the error and suggest a correction.
- Keep answers concise: lead with the direct answer, then show supporting data.
- Round currency to 2 decimal places and large numbers with commas for readability.""",
        tools=[
            load_csv,
            get_summary_stats,
            aggregate,
            top_n,
            count_values,
            filter_and_summarise,
        ],
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_agent()
        await agent.start()

        try:
            result = await agent.run(
                "Load data/sample_sales.csv and tell me the top 5 products by total revenue."
            )
            print(result)
        finally:
            await agent.stop()

    asyncio.run(main())
