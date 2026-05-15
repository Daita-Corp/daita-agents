"""Data-driven LLM pricing catalog.

Prices are USD per one million tokens. Keep this table intentionally small and
easy to update; unknown models fall back explicitly in the registry.
"""

from __future__ import annotations

PRICE_CATALOG = {
    "openai": {
        "gpt-5.5-pro*": {
            "input_per_1m": "30.00",
            "output_per_1m": "180.00",
            "source": "https://developers.openai.com/api/docs/models/compare",
        },
        "gpt-5.5*": {
            "input_per_1m": "5.00",
            "output_per_1m": "30.00",
            "cached_input_per_1m": "0.50",
            "source": "https://developers.openai.com/api/docs/models/gpt-5.5",
        },
        "gpt-5.4-pro*": {
            "input_per_1m": "30.00",
            "output_per_1m": "180.00",
            "source": "https://developers.openai.com/api/docs/models/compare",
        },
        "gpt-5.4-mini*": {
            "input_per_1m": "0.75",
            "output_per_1m": "4.50",
            "cached_input_per_1m": "0.075",
            "source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        },
        "gpt-5.4-nano*": {
            "input_per_1m": "0.20",
            "output_per_1m": "1.25",
            "cached_input_per_1m": "0.02",
            "source": "https://developers.openai.com/api/docs/models/gpt-5.4-nano",
        },
        "gpt-5.4*": {
            "input_per_1m": "2.50",
            "output_per_1m": "15.00",
            "cached_input_per_1m": "0.25",
            "source": "https://developers.openai.com/api/docs/models/gpt-5.4",
        },
        "gpt-5.2-pro*": {
            "input_per_1m": "21.00",
            "output_per_1m": "168.00",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-5.2*": {
            "input_per_1m": "1.75",
            "output_per_1m": "14.00",
            "cached_input_per_1m": "0.175",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-5.1*": {
            "input_per_1m": "1.25",
            "output_per_1m": "10.00",
            "cached_input_per_1m": "0.125",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-5-pro*": {
            "input_per_1m": "15.00",
            "output_per_1m": "120.00",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-5-mini*": {
            "input_per_1m": "0.25",
            "output_per_1m": "2.00",
            "cached_input_per_1m": "0.025",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-5-nano*": {
            "input_per_1m": "0.05",
            "output_per_1m": "0.40",
            "cached_input_per_1m": "0.005",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-5*": {
            "input_per_1m": "1.25",
            "output_per_1m": "10.00",
            "cached_input_per_1m": "0.125",
            "source": "https://developers.openai.com/api/docs/models/gpt-5",
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
        "o3-pro*": {
            "input_per_1m": "20.00",
            "output_per_1m": "80.00",
            "source": "https://openai.com/api/pricing/",
        },
        "o3-mini*": {
            "input_per_1m": "1.10",
            "output_per_1m": "4.40",
            "cached_input_per_1m": "0.55",
            "source": "https://openai.com/api/pricing/",
        },
        "o3*": {
            "input_per_1m": "2.00",
            "output_per_1m": "8.00",
            "cached_input_per_1m": "0.50",
            "source": "https://openai.com/api/pricing/",
        },
        "o4-mini*": {
            "input_per_1m": "1.10",
            "output_per_1m": "4.40",
            "cached_input_per_1m": "0.275",
            "source": "https://openai.com/api/pricing/",
        },
        "o1-pro*": {
            "input_per_1m": "150.00",
            "output_per_1m": "600.00",
            "source": "https://openai.com/api/pricing/",
        },
        "o1-mini*": {
            "input_per_1m": "1.10",
            "output_per_1m": "4.40",
            "cached_input_per_1m": "0.55",
            "source": "https://openai.com/api/pricing/",
        },
        "o1*": {
            "input_per_1m": "15.00",
            "output_per_1m": "60.00",
            "cached_input_per_1m": "7.50",
            "source": "https://openai.com/api/pricing/",
        },
        "gpt-3.5-turbo*": {
            "input_per_1m": "0.50",
            "output_per_1m": "1.50",
            "source": "https://openai.com/api/pricing/",
        },
    },
    "anthropic": {
        "claude-opus-4-7*": {
            "input_per_1m": "5.00",
            "output_per_1m": "25.00",
            "cached_input_per_1m": "0.50",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-opus-4-6*": {
            "input_per_1m": "5.00",
            "output_per_1m": "25.00",
            "cached_input_per_1m": "0.50",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-opus-4-5*": {
            "input_per_1m": "5.00",
            "output_per_1m": "25.00",
            "cached_input_per_1m": "0.50",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-opus-4-1*": {
            "input_per_1m": "15.00",
            "output_per_1m": "75.00",
            "cached_input_per_1m": "1.50",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-haiku-4-5*": {
            "input_per_1m": "1.00",
            "output_per_1m": "5.00",
            "cached_input_per_1m": "0.10",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-3-5-haiku*": {
            "input_per_1m": "0.80",
            "output_per_1m": "4.00",
            "cached_input_per_1m": "0.08",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-sonnet-4-6*": {
            "input_per_1m": "3.00",
            "output_per_1m": "15.00",
            "cached_input_per_1m": "0.30",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-sonnet-4-5*": {
            "input_per_1m": "3.00",
            "output_per_1m": "15.00",
            "cached_input_per_1m": "0.30",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-sonnet-4*": {
            "input_per_1m": "3.00",
            "output_per_1m": "15.00",
            "cached_input_per_1m": "0.30",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
        "claude-opus-4*": {
            "input_per_1m": "15.00",
            "output_per_1m": "75.00",
            "cached_input_per_1m": "1.50",
            "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        },
    },
    "gemini": {
        "gemini-3.1-pro-preview*": {
            "input_per_1m": "2.00",
            "output_per_1m": "12.00",
            "cached_input_per_1m": "0.20",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-3.1-flash-lite*": {
            "input_per_1m": "0.25",
            "output_per_1m": "1.50",
            "cached_input_per_1m": "0.025",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-3.1-flash*": {
            "input_per_1m": "0.50",
            "output_per_1m": "3.00",
            "cached_input_per_1m": "0.05",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-3-pro-image-preview*": {
            "input_per_1m": "2.00",
            "output_per_1m": "12.00",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-3-flash-preview*": {
            "input_per_1m": "0.50",
            "output_per_1m": "3.00",
            "cached_input_per_1m": "0.05",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-2.5-flash-lite*": {
            "input_per_1m": "0.10",
            "output_per_1m": "0.40",
            "cached_input_per_1m": "0.01",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-2.5-flash*": {
            "input_per_1m": "0.30",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.03",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "gemini-2.5-pro*": {
            "input_per_1m": "1.25",
            "output_per_1m": "10.00",
            "cached_input_per_1m": "0.125",
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
    },
    "grok": {
        "grok-4.3*": {
            "input_per_1m": "1.25",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.20",
            "source": "https://docs.x.ai/developers/models/grok-4.3",
        },
        "grok-latest": {
            "input_per_1m": "1.25",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.20",
            "source": "https://docs.x.ai/developers/models/grok-4.3",
        },
        "grok-4.20*": {
            "input_per_1m": "1.25",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.20",
            "source": "https://docs.x.ai/developers/models/grok-4.20",
        },
        "grok-code-fast-1*": {
            "input_per_1m": "1.25",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.20",
            "source": "https://docs.x.ai/developers/models",
        },
        "grok-4-1-fast*": {
            "input_per_1m": "1.25",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.20",
            "source": "https://docs.x.ai/developers/models",
        },
        "grok-4-fast*": {
            "input_per_1m": "1.25",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.20",
            "source": "https://docs.x.ai/developers/models",
        },
        "grok-4*": {
            "input_per_1m": "1.25",
            "output_per_1m": "2.50",
            "cached_input_per_1m": "0.20",
            "source": "https://docs.x.ai/developers/models",
        },
    },
}

FALLBACK_PRICING = {
    "input_per_1m": "2.00",
    "output_per_1m": "2.00",
    "source": "fallback:generic-flat-rate",
}
