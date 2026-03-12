"""
Transformer Agent

Receives raw event records, validates and normalises them using pandas,
deduplicates, and emits cleaned records to the transformed_data channel.
"""

import json
from typing import Any, Dict, List

from daita import Agent
from daita.core.tools import tool


@tool
async def validate_and_clean(records_json: str) -> Dict[str, Any]:
    """
    Validate and clean a batch of raw event records.

    Transformation rules applied:
    - Drop records missing user_id or event_type
    - Normalise event_type to lowercase with underscores
    - Parse properties from JSON string to dict (if not already)
    - Deduplicate by (user_id, event_type, session_id, created_at)
    - Add a processed_at timestamp

    Args:
        records_json: JSON string containing a list of event dicts.

    Returns:
        Dict with `clean` (list), `rejected` (list with reasons), and `stats`.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required. Install with: pip install 'daita-agents[data]'"
        )

    from datetime import datetime, timezone

    try:
        records = json.loads(records_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid JSON input: {e}"}

    if not records:
        return {
            "clean": [],
            "rejected": [],
            "stats": {"input": 0, "clean": 0, "rejected": 0, "duplicates_removed": 0},
        }

    df = pd.DataFrame(records)
    original_count = len(df)
    rejected: List[Dict[str, Any]] = []

    # Drop records missing required fields
    for field in ["user_id", "event_type"]:
        if field in df.columns:
            mask = df[field].isna() | (df[field].astype(str).str.strip() == "")
            bad_rows = df[mask].copy()
            bad_rows["rejection_reason"] = f"missing_{field}"
            rejected.extend(bad_rows.to_dict(orient="records"))
            df = df[~mask]

    if df.empty:
        return {
            "clean": [],
            "rejected": rejected,
            "stats": {
                "input": original_count,
                "clean": 0,
                "rejected": len(rejected),
                "duplicates_removed": 0,
            },
        }

    # Normalise event_type
    df["event_type"] = (
        df["event_type"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Parse properties if it's a string
    def parse_properties(val: Any) -> Any:
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                return {}
        return val if isinstance(val, dict) else {}

    if "properties" in df.columns:
        df["properties"] = df["properties"].apply(parse_properties)

    # Deduplicate
    dedup_cols = [c for c in ["user_id", "event_type", "session_id", "created_at"] if c in df.columns]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    duplicates_removed = before_dedup - len(df)

    # Add processed_at
    df["processed_at"] = datetime.now(timezone.utc).isoformat()

    clean_records = df.to_dict(orient="records")

    return {
        "clean": clean_records,
        "rejected": rejected,
        "stats": {
            "input": original_count,
            "clean": len(clean_records),
            "rejected": len(rejected),
            "duplicates_removed": duplicates_removed,
        },
    }


def create_agent() -> Agent:
    """Create the Transformer agent."""
    return Agent(
        name="Transformer",
        model="gpt-4o-mini",
        prompt="""You clean and normalise raw event records passed from the Extractor.

Your steps:
1. Receive the raw records JSON from the relay channel.
2. Call validate_and_clean with the records JSON to apply transformation rules.
3. Report the stats: how many records were cleaned, rejected, and deduplicated.
4. If any records were rejected, list the reasons.
5. Pass the clean records downstream for loading.

Transformation rules applied by the tool:
- Drop records missing user_id or event_type
- Normalise event_type to lowercase_underscore format
- Parse properties from JSON string to dict
- Deduplicate by (user_id, event_type, session_id, created_at)
- Add processed_at timestamp

If stats.clean is 0, report that no records passed validation and stop.""",
        tools=[validate_and_clean],
    )
