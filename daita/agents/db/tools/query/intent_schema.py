"""Shared JSON schemas for query intent tools."""

from __future__ import annotations

from typing import Any, Dict


def query_intent_parameters(
    *, include_diagnostics: bool = False, mode: str = "plan"
) -> Dict[str, Any]:
    properties = {
        "goal": {
            "type": "string",
            "description": "Plain-English objective for the query.",
        },
        "required_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fields or metrics the final answer must include.",
        },
        "candidate_tables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Likely tables or collections involved.",
        },
        "filters": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Natural-language filters or SQL-safe clauses.",
        },
        "aggregations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Requested metrics such as count, sum, average.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum rows requested, if any.",
        },
    }
    if mode == "plan":
        properties.update(
            {
                "required_joins": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Join requirements with from_tables and to_tables arrays."
                    ),
                },
                "grouping": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to group by.",
                },
                "ordering": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sort requirements.",
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Assumptions needed to proceed.",
                },
                "answer_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Completeness checks for the final answer.",
                },
            }
        )
    elif mode != "compile":
        raise ValueError("mode must be 'plan' or 'compile'")
    if include_diagnostics:
        properties["include_diagnostics"] = {
            "type": "boolean",
            "description": (
                "Return full candidate tables, field candidates, join paths, "
                "and query IR diagnostics. Default false."
            ),
        }
    return {"type": "object", "properties": properties, "required": ["goal"]}
