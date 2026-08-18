"""Thin lazy entry point for Daita's one Textual application."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from ._installation import repair_guidance
from .security import KeychainStore
from .tui.models import validate_candidate_review_cost_limit


def _load_textual_app() -> Callable[..., Any]:
    try:
        import textual
    except ImportError as error:
        raise ImportError(
            "Daita's terminal UI is unavailable. " + repair_guidance()
        ) from error
    del textual
    from .tui.app import run_daita_app

    return run_daita_app


async def run_terminal_application(
    *,
    root: str | Path | None = None,
    agent_name: str | None = None,
    reviewer_max_estimated_cost_usd: Decimal | None = None,
    keychain: KeychainStore | None = None,
    model_validator: Any = None,
    input_stream: Any = None,
    output_stream: Any = None,
    hidden_input: Any = None,
    selection_input: Any = None,
    selection_output: Any = None,
    tui_input: Any = None,
    tui_output: Any = None,
) -> int:
    """Start one Textual app. Extra stream arguments are ignored after cutover."""

    validate_candidate_review_cost_limit(reviewer_max_estimated_cost_usd)
    run_app = _load_textual_app()
    return await run_app(
        root=root,
        agent_name=agent_name,
        keychain=keychain,
        model_validator=model_validator,
        reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
    )


__all__ = ["run_terminal_application"]
