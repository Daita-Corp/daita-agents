"""Versioned natural-language prompt corpus for LIVE-MVP-01 through 04."""

from __future__ import annotations

PROMPT_CORPUS_VERSION = "wave1-prompts-v1"

GROUNDING_PROMPTS = (
    (
        "How many active customers placed paid orders from January 1 through "
        "March 31, 2026, and what was their net revenue after refunds?"
    ),
    (
        "Could you work out, for Q1 2026, how many currently active customers "
        "actually paid for an order and how much revenue remained after refunds?"
    ),
    (
        "For the first reporting quarter of 2026, give me the active-customer "
        "count and net paid revenue after any refunds."
    ),
)

CATALOG_GRAPH_PROMPTS = (
    (
        "Which region had the highest net revenue from active customers during "
        "January 1 through March 31, 2026, and what was the amount?"
    ),
    (
        "For Q1 2026, can you tell me which current customer region led on net "
        "paid revenue after refunds? Include the amount."
    ),
    (
        "What region led current active-customer revenue in the first reporting "
        "quarter of 2026 once refunds are taken out?"
    ),
)

CROSS_SOURCE_PROMPTS = (
    "Compare the newest customer export with our current customer data and explain every discrepancy.",
    (
        "Could you check the latest customer export against the customer records "
        "we use now and walk me through all differences?"
    ),
    (
        "Reconcile the freshest customer export with current customer information; "
        "I need every mismatch or one-sided record explained."
    ),
)

SESSION_INITIAL_PROMPTS = GROUNDING_PROMPTS

SESSION_FOLLOW_UPS = (
    "Break that down by customer plan.",
    "Only show enterprise customers.",
    "Within that enterprise result, break it down by region.",
    "Which region contributed the most to that enterprise figure?",
    "Restate the enterprise total and its leading region in one sentence.",
)
MVP_SESSION_FOLLOW_UPS = (SESSION_FOLLOW_UPS[1],)

SESSION_POST_REOPEN_PROMPT = (
    "How many of those enterprise customers received refunds during that same "
    "period, and how much was refunded in total?"
)

PROMPTS_BY_SCENARIO = {
    "LIVE-MVP-01": GROUNDING_PROMPTS,
    "LIVE-MVP-02": CATALOG_GRAPH_PROMPTS,
    "LIVE-MVP-03": CROSS_SOURCE_PROMPTS,
    "LIVE-MVP-04": SESSION_INITIAL_PROMPTS,
}


__all__ = [
    "CATALOG_GRAPH_PROMPTS",
    "CROSS_SOURCE_PROMPTS",
    "GROUNDING_PROMPTS",
    "MVP_SESSION_FOLLOW_UPS",
    "PROMPTS_BY_SCENARIO",
    "PROMPT_CORPUS_VERSION",
    "SESSION_FOLLOW_UPS",
    "SESSION_INITIAL_PROMPTS",
    "SESSION_POST_REOPEN_PROMPT",
]
