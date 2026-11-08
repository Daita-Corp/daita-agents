"""Scoring and Claude Code runner shared by the live Daita-vs-Claude Code harnesses.

Importing this module contacts nothing and resolves no credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pytest

ANSWER_FORMAT = " Finish with the final answer as a single value on the last line."
OUTCOMES = ("correct", "confident_wrong", "no_answer", "incomplete")
# shortcut: keyword heuristic for live answers, replace with labeled judgments
# once the head-to-head comparison needs per-answer human review.
REFUSAL = re.compile(
    r"\b(cannot|can't|can not|unable|not available|isn't available|"
    r"not (?:in|part of) the catalog|no access|don't have access|not cataloged|"
    r"not accessible|couldn't|could not|not found|no answer|n/a)\b",
    re.IGNORECASE,
)
CLARIFICATION = re.compile(
    r"\b(clarify|could you confirm|can you confirm|please confirm|please specify|"
    r"which (?:one|table)|did you mean|do you mean)\b",
    re.IGNORECASE,
)
# A run that could not use its tools says so; that is a decline, not a wrong value.
PERMISSION = re.compile(
    r"\b(?:permission|denied|don'?t ask|bash access|could you (?:run|execute))\b",
    re.IGNORECASE,
)


def normalize(value: object) -> str:
    if isinstance(value, Mapping) and "value" in value:
        value = value["value"]  # Tagged JSON projection, e.g. decimals.
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return str(value).strip()
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def answer_in_text(expected: str, text: str) -> bool:
    flattened = text.replace(",", "").replace("$", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", expected):
        candidates = {expected}
        if "." in expected:
            candidates.add(f"{Decimal(expected):.2f}")
        else:
            candidates.add(f"{expected}.00")
        return any(
            re.search(rf"(?<![\d.]){re.escape(item)}(?!\d|\.\d)", flattened)
            for item in candidates
        )
    return re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", text, re.I) is not None


def last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def classify(expected: str, text: str, completed: bool) -> str:
    """Score Daita and Claude Code answers with one rule.

    ``incomplete`` is a run stopped by a limit or an error; ``no_answer`` is an
    honest decline or clarification request; only the requested last line can
    be correct or confidently wrong.
    """

    last = last_line(text)
    if not completed or not last:
        return "incomplete"
    if answer_in_text(expected, last):
        return "correct"
    declines = (REFUSAL, CLARIFICATION, PERMISSION)
    if any(item.search(last) for item in declines) or (
        not re.search(r"\d", last) and any(item.search(text) for item in declines)
    ):
        return "no_answer"
    return "confident_wrong"


def answer_record(expected: str, text: str, completed: bool) -> dict[str, object]:
    return {
        "outcome": classify(expected, text, completed),
        "expected": expected,
        "final_text": text[:2_000],
        # Audit only: the value appears somewhere but not on the scored line.
        "answer_elsewhere": not answer_in_text(expected, last_line(text))
        and answer_in_text(expected, text),
        "asked_clarification": CLARIFICATION.search(text) is not None,
    }


def system_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    count = max(len(runs), 1)
    costs = [run["cost_usd"] for run in runs]
    turns = sum(int(run["turns"]) for run in runs)
    return {
        "runs": len(runs),
        "outcomes": {
            name: sum(1 for run in runs if run["outcome"] == name) for name in OUTCOMES
        },
        "answer_elsewhere": sum(1 for run in runs if run["answer_elsewhere"]),
        "mean_run_seconds": round(
            sum(float(run["run_seconds"]) for run in runs) / count, 1
        ),
        "total_turns": turns,
        "total_tool_calls": sum(int(run["tool_calls"]) for run in runs),
        "total_tokens": sum(int(run["total_tokens"]) for run in runs),
        "mean_input_tokens_per_turn": round(
            sum(int(run["input_tokens"]) for run in runs) / max(turns, 1)
        ),
        "total_cost_usd": (
            None
            if None in costs
            else str(round(sum(Decimal(str(c)) for c in costs), 4))
        ),
    }


# Traps where a catalog fact (comment, note, statistic, stale note, injected
# instruction) can make an answer worse than having no fact at all.
HARM_TRAPS = frozenset(
    {
        "look_alike_comment",
        "look_alike_note",
        "look_alike_stats",
        "stale_note",
        "injection",
    }
)


def hard_record(
    question: Mapping[str, Any],
    expected: str | None,
    trap_value: str | None,
    text: str,
    completed: bool,
) -> dict[str, object]:
    """Score a hard-tier answer and note whether it fell for the trap.

    A ``decline_expected`` question has no value (``expected`` is None): the
    right answer is a decline, anything else that finishes is confidently wrong.
    """

    if question.get("decline_expected"):
        outcome = classify("\0", text, completed)  # No value can match.
        record: dict[str, object] = {
            "outcome": (
                "correct"
                if outcome == "no_answer"
                else outcome if outcome == "incomplete" else "confident_wrong"
            ),
            "expected": None,
            "final_text": text[:2_000],
            "answer_elsewhere": False,
            "asked_clarification": CLARIFICATION.search(text) is not None,
            "decline_expected": True,
        }
        return record | {
            "trap": question["trap"],
            "trap_value": None,
            "trap_followed": False,
            "evidence_clear": False,
        }
    record = answer_record(str(expected), text, completed)
    return record | {
        "trap": question["trap"],
        "trap_value": trap_value,
        "trap_followed": (
            trap_value is not None
            and record["outcome"] != "correct"
            and answer_in_text(trap_value, last_line(text))
        ),
        "evidence_clear": bool(question.get("evidence_clear")),
    }


def schema_change_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Correct count per schema-change question, plus failed SQL calls and turns."""

    chosen = [run for run in runs if run["category"] == "schema_change"]
    questions: dict[str, dict[str, int]] = {}
    for run in chosen:
        item = questions.setdefault(str(run["id"]), {"correct": 0, "runs": 0})
        item["runs"] += 1
        item["correct"] += run["outcome"] == "correct"
    return {
        "questions": questions,
        # Claude Code records carry no failed_sql_calls.
        "failed_sql_calls": sum(int(run.get("failed_sql_calls", 0)) for run in chosen),
        "turns": sum(int(run["turns"]) for run in chosen),
    }


def hard_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Hard-tier totals; ``harm`` counts answers a catalog fact made worse."""

    def tally(chosen: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "correct": sum(1 for run in chosen if run["outcome"] == "correct"),
            "runs": len(chosen),
        }

    clear = [run for run in runs if run["evidence_clear"]]
    return {
        "outcomes": {
            name: sum(1 for run in runs if run["outcome"] == name) for name in OUTCOMES
        },
        "harm": sum(
            1 for run in runs if run["trap_followed"] and run["trap"] in HARM_TRAPS
        ),
        "fan_out_followed": sum(
            1 for run in runs if run["trap_followed"] and run["trap"] == "fan_out"
        ),
        "look_alike_resolved": {
            "resolved": sum(
                1
                for run in clear
                if run["outcome"] == "correct" and not run["asked_clarification"]
            ),
            "runs": len(clear),
        },
        "correction_carried": tally(
            [run for run in runs if str(run["id"]).startswith("h10")]
        ),
        "follow_up_correct": tally(
            [run for run in runs if str(run["id"]).endswith("-follow-up")]
        ),
        "schema_change": schema_change_summary(runs),
    }


async def ask_claude_code(
    executable: str,
    model: str,
    prompt: str,
    *,
    environment: Mapping[str, str],
    allowed_tools: str,
    max_cost: str,
    workdir: Path,
    config: Path,
    tools: str = "Bash",
    persist_session: bool = False,
    resume: str | None = None,
) -> dict[str, Any]:
    """Run ``claude -p`` once with ``tools`` limited by ``allowed_tools``.

    ``persist_session`` keeps the session so a later call can ``resume`` it; the
    run record then carries its ``session_id``.
    """

    env = {
        key: value
        for key, value in os.environ.items()
        # A fresh top-level session: no parent-session settings such as
        # CLAUDE_EFFORT, and no harness configuration.
        if not key.startswith(("CLAUDE", "DAITA_"))
    } | {
        **environment,
        # Empty config: no user CLAUDE.md, plugins, hooks, memory or login, while
        # keeping Claude Code's default prompt and tools (unlike --bare).
        "CLAUDE_CONFIG_DIR": str(config),
    }
    started = time.perf_counter()
    # An empty working directory and a narrow Bash allowlist: Claude Code sees
    # the data and nothing in this repository, which holds the answer SQL.
    process = await asyncio.create_subprocess_exec(
        executable,
        "-p",
        prompt,
        *(("--resume", resume) if resume else ()),
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--tools",
        tools,
        "--allowedTools",
        allowed_tools,
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        *(() if persist_session else ("--no-session-persistence",)),
        "--max-budget-usd",
        max_cost,
        cwd=workdir,
        env=env,
        # claude -p appends piped stdin to the prompt; never inherit the caller's.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(600):
            stdout, stderr = await process.communicate()
    except TimeoutError:
        process.kill()
        await process.wait()
        stdout, stderr = b"", b"timed out after 600 seconds"
    events = []
    for line in stdout.decode(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    commands = [
        str(block.get("input", {}).get("command", ""))[:1_000]
        for event in events
        if event.get("type") == "assistant"
        for block in event.get("message", {}).get("content", ())
        if block.get("type") == "tool_use"
    ]
    # Tool results flagged as errors; a denial names the permission in its text.
    failed = [
        str(block.get("content"))
        for event in events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", ())
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("is_error")
    ]
    denied = [text for text in failed if PERMISSION.search(text)]
    last_result = next(
        (
            block
            for event in reversed(events)
            if event.get("type") == "user"
            for block in event.get("message", {}).get("content", ())
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ),
        None,
    )
    result: dict[str, Any] = next(
        (event for event in reversed(events) if event.get("type") == "result"), {}
    )
    usage = result.get("usage") or {}
    completed = result.get("subtype") == "success" and not result.get("is_error")
    input_tokens = sum(
        int(usage.get(key) or 0)
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )
    return {
        "completed": completed,
        "final_text": str(result.get("result") or "") if completed else "",
        "run_seconds": round(time.perf_counter() - started, 3),
        "turns": int(result.get("num_turns") or 0),
        "tool_calls": len(commands),
        "commands": commands,
        "errored_calls": len(failed),
        "denied_calls": max(len(denied), len(result.get("permission_denials") or ())),
        # The last tool result was a denial: the run stopped on it.
        "ended_on_denial": bool(
            last_result
            and last_result.get("is_error")
            and PERMISSION.search(str(last_result.get("content")))
        ),
        "input_tokens": input_tokens,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "total_tokens": input_tokens + int(usage.get("output_tokens") or 0),
        "cost_usd": result.get("total_cost_usd"),
        "session_id": result.get("session_id"),
        "exit": result.get("subtype") or f"exit {process.returncode}",
        "api_error_status": result.get("api_error_status"),
        "error": (
            str(result.get("result"))[:500]
            if result.get("is_error")
            else stderr.decode(errors="replace")[-500:] if not result else None
        ),
    }


def claude_executable(executable_env: str) -> str | None:
    return os.environ.get(executable_env) or shutil.which("claude")


def require_claude(executable_env: str) -> str:
    executable = claude_executable(executable_env)
    if executable is None:
        pytest.fail(f"claude is not on PATH; set {executable_env}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail(
            "Claude Code runs with an empty config, so it needs ANTHROPIC_API_KEY"
        )
    return executable


def fail_on_api_rejection(run: Mapping[str, Any]) -> None:
    # Auth, billing and other API rejections are setup failures, not answers.
    if run["api_error_status"] or (
        run["error"] and "authenticate" in run["error"].lower()
    ):
        pytest.fail(f"Claude Code could not call the model: {run['error']}")


async def run_claude_code(
    model: str,
    questions: Sequence[Mapping[str, Any]],
    truth: Mapping[str, str],
    root: Path,
    entry: dict[str, Any],
    repeat: int,
    *,
    executable_env: str,
    preamble: str,
    environment: Mapping[str, str],
    allowed_tools: str,
    max_cost: str,
) -> None:
    """Ask every question once; append scored runs to ``entry["questions"]``."""

    executable = require_claude(executable_env)
    for question in questions:
        workdir = root / question["id"]
        config = root / f"{question['id']}-config"
        workdir.mkdir(parents=True)
        config.mkdir()
        run = await ask_claude_code(
            executable,
            model,
            preamble + question["question"] + ANSWER_FORMAT,
            environment=environment,
            allowed_tools=allowed_tools,
            max_cost=max_cost,
            workdir=workdir,
            config=config,
        )
        fail_on_api_rejection(run)
        run.pop("session_id")
        completed = run.pop("completed")
        text = run.pop("final_text")
        entry["questions"].append(
            {
                "id": question["id"],
                "category": question["category"],
                "repeat": repeat,
                **answer_record(truth[question["id"]], text, completed),
                **run,
            }
        )


def claude_code_version(executable_env: str) -> str | None:
    executable = claude_executable(executable_env)
    if executable is None:
        return None
    return subprocess.run(
        (executable, "--version"), capture_output=True, text=True, check=False
    ).stdout.strip()
