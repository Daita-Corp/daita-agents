"""Natural-language Stage B benchmark without model-facing tool instructions.

The complete module authorizes at most six ``Agent.run`` interactions.
Every interaction retains the shared step, token, wall-time, and estimated-cost
ceilings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest

from tests.support.job_benchmarks import (
    DEFAULT_MODEL_ID,
    IMMEDIATE_AMOUNT,
    IMMEDIATE_TOKEN,
    PROFILE_NULL_VALUES,
    PROFILE_SAMPLE_ROWS,
    TARGET_IMMEDIATE_TABLE,
    TARGET_PROFILE_TABLE,
    assert_completed,
    benchmark_marks,
    capture_run,
    create_live_agent,
    logical_names,
    record_metrics,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_B_PARAPHRASE_BENCHMARK"
LIFECYCLE_TOOLS = frozenset({"job_start", "job_list", "job_inspect", "job_cancel"})

pytestmark = benchmark_marks(_AUTHORIZATION, maximum_interactions=6)

_IMMEDIATE_PROMPTS = (
    (
        "quick-lookup",
        "Could you quickly look up the row named immediate in "
        "stage_b_immediate_probe and tell me its verification token and amount? "
        "I only need the answer now; do not launch background work.",
    ),
    (
        "plain-language",
        "What token and amount are stored on the immediate record in "
        "stage_b_immediate_probe? This is just a small lookup, not a job.",
    ),
    (
        "negative-instruction",
        "Please check stage_b_immediate_probe for probe_name immediate. Answer "
        "from the current data and don't queue or start anything long-running.",
    ),
)

_PROFILE_PROMPTS = (
    (
        "profile-summary",
        "Please profile stage_b_profile_probe: how many rows does it have, and "
        "how many nullable_code values are null?",
    ),
    (
        "data-quality",
        "Give me a quick data-quality check of stage_b_profile_probe: the row "
        "count and the number of missing nullable_code values.",
    ),
    (
        "background-wording",
        "Profile stage_b_profile_probe in the background if you need to, but I "
        "only want its total row count and nullable_code null count.",
    ),
)


@pytest.mark.parametrize(
    ("case_id", "prompt"),
    _IMMEDIATE_PROMPTS,
    ids=tuple(item[0] for item in _IMMEDIATE_PROMPTS),
)
async def test_paraphrased_immediate_reads_do_not_create_jobs(
    tmp_path: Path,
    case_id: str,
    prompt: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(
        tmp_path,
        f"paraphrase-immediate-{case_id}",
        DEFAULT_MODEL_ID,
    )
    try:
        capture = await capture_run(
            fixture,
            prompt,
            source_id=fixture.source_id,
        )
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        names = logical_names(capture.transcript)
        assert "data_query" in names
        assert await fixture.agent.list_jobs() == ()
        assert capture.result.final_text is not None
        assert IMMEDIATE_TOKEN in capture.result.final_text
        assert str(IMMEDIATE_AMOUNT) in capture.result.final_text
    finally:
        await fixture.close()


@pytest.mark.parametrize(
    ("case_id", "prompt"),
    _PROFILE_PROMPTS,
    ids=tuple(item[0] for item in _PROFILE_PROMPTS),
)
async def test_paraphrased_profile_requests_use_aggregates_without_jobs(
    tmp_path: Path,
    case_id: str,
    prompt: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(
        tmp_path,
        f"paraphrase-profile-{case_id}",
        DEFAULT_MODEL_ID,
    )
    try:
        capture = await capture_run(
            fixture,
            prompt,
            source_id=fixture.source_id,
        )
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        names = logical_names(capture.transcript)
        assert "data_query" in names
        assert not (LIFECYCLE_TOOLS & set(names))
        assert await fixture.agent.list_jobs() == ()
        target = fixture.resource_ids[TARGET_PROFILE_TABLE]
        assert any(
            target in cast(Sequence[str], call.arguments.get("resource_ids", ()))
            for call, _ in capture.evidence
            if call.name == "data_query"
        )
        assert capture.result.final_text is not None
        assert str(PROFILE_SAMPLE_ROWS) in capture.result.final_text
        assert str(PROFILE_NULL_VALUES) in capture.result.final_text
    finally:
        await fixture.close()


def test_paraphrase_case_inventory_is_bounded_and_nonduplicative() -> None:
    cases = (
        *_IMMEDIATE_PROMPTS,
        *_PROFILE_PROMPTS,
    )
    prompts = tuple(prompt for _, prompt in cases)
    assert len(cases) == 6
    assert len(prompts) == len(set(prompts))
    assert all(20 <= len(prompt) <= 600 for prompt in prompts)
    assert all(
        forbidden not in prompt
        for prompt in prompts
        for forbidden in (
            "toolbox_search",
            "toolbox_load",
            "data_query",
            "job_list",
            "job_start",
            "job_cancel",
        )
    )
    assert TARGET_IMMEDIATE_TABLE in " ".join(prompts)
