"""Data-driven LLM pricing catalog.

Prices are USD per one million tokens. Keep this table intentionally small and
easy to update; unknown models fall back explicitly in the registry.
"""

from __future__ import annotations

PRICE_CATALOG = {
    "openai": {
        "gpt-4o-mini*": {
            "input_per_1m": "0.15",
            "output_per_1m": "0.60",
            "cached_input_per_1m": "0.075",
            "source": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        },
        "gpt-4o*": {
            "input_per_1m": "2.50",
            "output_per_1m": "10.00",
            "cached_input_per_1m": "1.25",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-4.1-mini*": {
            "input_per_1m": "0.40",
            "output_per_1m": "1.60",
            "cached_input_per_1m": "0.10",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-4.1-nano*": {
            "input_per_1m": "0.10",
            "output_per_1m": "0.40",
            "cached_input_per_1m": "0.025",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-4.1*": {
            "input_per_1m": "2.00",
            "output_per_1m": "8.00",
            "cached_input_per_1m": "0.50",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-3.5-turbo*": {
            "input_per_1m": "0.50",
            "output_per_1m": "1.50",
            "source": "https://openai.com/api/pricing/",
        },
    },
    "anthropic": {
        "claude-haiku-4-5*": {
            "input_per_1m": "1.00",
            "output_per_1m": "5.00",
            "cached_input_per_1m": "0.10",
            "source": "https://www.anthropic.com/pricing",
        },
        "claude-3-5-haiku*": {
            "input_per_1m": "0.80",
            "output_per_1m": "4.00",
            "cached_input_per_1m": "0.08",
            "source": "https://www.anthropic.com/pricing",
        },
        "claude-sonnet-4*": {
            "input_per_1m": "3.00",
            "output_per_1m": "15.00",
            "cached_input_per_1m": "0.30",
            "source": "https://www.anthropic.com/pricing",
        },
        "claude-opus-4*": {
            "input_per_1m": "15.00",
            "output_per_1m": "75.00",
            "cached_input_per_1m": "1.50",
            "source": "https://www.anthropic.com/pricing",
        },
    },
    "gemini": {
        "gemini-2.5-flash-lite*": {
            "input_per_1m": "0.10",
            "output_per_1m": "0.40",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-2.5-flash*": {
            "input_per_1m": "0.30",
            "output_per_1m": "2.50",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-2.5-pro*": {
            "input_per_1m": "1.25",
            "output_per_1m": "10.00",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
    },
    "grok": {
        "grok-4.20*": {
            "input_per_1m": "0.20",
            "output_per_1m": "0.50",
            "source": "https://docs.x.ai/docs/models",
        },
        "grok-4*": {
            "input_per_1m": "3.00",
            "output_per_1m": "15.00",
            "source": "https://docs.x.ai/docs/models",
        },
    },
}

FALLBACK_PRICING = {
    "input_per_1m": "2.00",
    "output_per_1m": "2.00",
    "source": "fallback:generic-flat-rate",
}

