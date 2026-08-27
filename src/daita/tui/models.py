"""Define framework-neutral records and validation for the terminal presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

MAX_COMPOSER_CHARACTERS = 16_384
MIN_USABLE_COLUMNS = 32
MIN_READY_ROWS = 8
MIN_APPROVAL_ROWS = 15
MAX_QUEUED_EVENTS = 4_096

PROVIDERS = (
    ("openai", "OpenAI API"),
    ("anthropic", "Anthropic API"),
    ("gemini", "Gemini API"),
    ("grok", "xAI (Grok) API"),
    ("ollama", "Ollama local"),
    ("codex", "Codex subscription"),
    ("claude-code", "Claude Code subscription"),
    ("grok-build", "Grok Build subscription"),
    ("custom", "Custom API (OpenAI-compatible)"),
)
BUILTIN_PROVIDER_IDS = frozenset(provider for provider, _ in PROVIDERS[:-1])
SUBSCRIPTION_PROVIDER_IDS = frozenset({"codex", "claude-code", "grok-build"})
SOURCE_TYPES = (
    ("sqlite", "SQLite file"),
    ("postgresql", "PostgreSQL"),
)
SOURCE_TYPE_LABELS = {
    "postgresql": "PostgreSQL",
    "sqlite": "SQLite",
}
SSL_MODES = frozenset(
    {"disable", "prefer", "allow", "require", "verify-ca", "verify-full"}
)
MAX_POSTGRESQL_CONNECTION_URL_BYTES = 64 * 1_024
POSTGRESQL_CONNECTION_URL_ERROR = (
    "Enter a valid postgres:// or postgresql:// URL with a username, host, "
    "and database. Only the sslmode query parameter is supported."
)
DEFAULT_CANDIDATE_REVIEW_COST_LIMIT_USD = Decimal("0.05")


@dataclass(frozen=True, slots=True)
class ModelSuggestion:
    provider_id: str
    model_id: str
    label: str
    description: str
    recommendation: str | None = None


MODEL_SUGGESTIONS = {
    "openai": (
        ModelSuggestion(
            "openai",
            "gpt-5.6-sol",
            "GPT-5.6 Sol",
            "Frontier capability for complex data-agent work",
            "Recommended",
        ),
        ModelSuggestion(
            "openai",
            "gpt-5.6-terra",
            "GPT-5.6 Terra",
            "Balanced intelligence and cost for everyday workflows",
            "Balanced",
        ),
        ModelSuggestion(
            "openai",
            "gpt-5.6-luna",
            "GPT-5.6 Luna",
            "Efficient model for high-volume bounded tasks",
            "Fast",
        ),
    ),
    "anthropic": (
        ModelSuggestion(
            "anthropic",
            "claude-opus-4-8",
            "Claude Opus 4.8",
            "Complex agentic and enterprise work",
            "Strong",
        ),
        ModelSuggestion(
            "anthropic",
            "claude-sonnet-5",
            "Claude Sonnet 5",
            "Fast balance of speed and intelligence",
            "Recommended",
        ),
        ModelSuggestion(
            "anthropic",
            "claude-haiku-4-5-20251001",
            "Claude Haiku 4.5",
            "Low-latency near-frontier model",
            "Fast",
        ),
    ),
    "codex": (
        ModelSuggestion(
            "codex",
            "gpt-5.6-sol",
            "GPT-5.6 Sol",
            "Connect ChatGPT to Daita and use the subscription allowance",
            "Recommended",
        ),
        ModelSuggestion(
            "codex",
            "gpt-5.6-terra",
            "GPT-5.6 Terra",
            "Balanced Codex model through Daita's ChatGPT connection",
            "Balanced",
        ),
        ModelSuggestion(
            "codex",
            "gpt-5.6-luna",
            "GPT-5.6 Luna",
            "Efficient Codex model through Daita's ChatGPT connection",
            "Fast",
        ),
    ),
    "claude-code": (
        ModelSuggestion(
            "claude-code",
            "claude-sonnet-5",
            "Claude Sonnet 5",
            "Use Claude through an existing Claude Code subscription login",
            "Recommended",
        ),
        ModelSuggestion(
            "claude-code",
            "claude-opus-4-8",
            "Claude Opus 4.8",
            "Complex work through the signed-in Claude Code client",
            "Strong",
        ),
        ModelSuggestion(
            "claude-code",
            "claude-haiku-4-5-20251001",
            "Claude Haiku 4.5",
            "Low-latency work through the signed-in Claude Code client",
            "Fast",
        ),
    ),
    "grok-build": (
        ModelSuggestion(
            "grok-build",
            "grok-4.5",
            "Grok 4.5",
            "Use Grok 4.5 through an existing Grok Build subscription login",
            "Recommended",
        ),
    ),
    "gemini": (
        ModelSuggestion(
            "gemini",
            "gemini-3.6-flash",
            "Gemini 3.6 Flash",
            "Stable agentic model balancing speed and intelligence",
            "Recommended",
        ),
        ModelSuggestion(
            "gemini",
            "gemini-3.5-flash",
            "Gemini 3.5 Flash",
            "Sustained performance for long-running agent work",
            "Strong",
        ),
        ModelSuggestion(
            "gemini",
            "gemini-3.5-flash-lite",
            "Gemini 3.5 Flash-Lite",
            "Low-latency model for high-volume agent tasks",
            "Fast",
        ),
    ),
    "grok": (
        ModelSuggestion(
            "grok",
            "grok-4.5",
            "Grok 4.5",
            "Agentic tool calling for general and code workflows",
            "Recommended",
        ),
    ),
    "ollama": (
        ModelSuggestion(
            "ollama",
            "qwen3",
            "Qwen 3",
            "Common local model with tool and reasoning support",
            "Recommended",
        ),
        ModelSuggestion(
            "ollama",
            "llama3.1",
            "Llama 3.1",
            "Common local model with tool-use support",
            "Lightweight",
        ),
        ModelSuggestion(
            "ollama",
            "mistral-small3.2",
            "Mistral Small 3.2",
            "Local model improved for function calling",
            "Tool use",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class PickerOption:
    identity: str
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    kind: str
    message: str = ""
    conversation_id: str | None = None
    run_message: str | None = None
    source_id: str | None = None
    screen: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolTablePreview:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    recorded_rows: int
    recorded_columns: int
    total_rows: int | None = None
    cells_truncated: bool = False


@dataclass(frozen=True, slots=True)
class ToolCardDetails:
    summary: str
    code: str | None = None
    code_language: str | None = None
    arguments_text: str | None = None
    result_text: str | None = None
    error_message: str | None = None
    table: ToolTablePreview | None = None


@dataclass(slots=True)
class ToolCardState:
    run_id: str
    call_id: str
    capability_id: str | None
    label: str
    state: str = "queued"
    duration_ms: int | None = None
    error_code: str | None = None
    approval_outcome: str | None = None
    details: ToolCardDetails | None = None
    expanded: bool = False


@dataclass(slots=True)
class TranscriptBlock:
    kind: str
    identity: str
    text: str = ""
    tool_card: ToolCardState | None = None


class UserInputError(ValueError):
    """Recoverable composer or command input that should not close the app."""


def validate_candidate_review_cost_limit(value: Decimal | None) -> None:
    if value is not None and (
        not isinstance(value, Decimal) or not value.is_finite() or value < 0
    ):
        raise ValueError(
            "candidate review cost ceiling must be finite and non-negative"
        )


def parse_candidate_review_cost_limit(value: str) -> Decimal:
    try:
        limit = Decimal(value)
    except Exception:
        raise ValueError(
            "candidate review cost ceiling must be a finite non-negative decimal"
        ) from None
    validate_candidate_review_cost_limit(limit)
    return limit
