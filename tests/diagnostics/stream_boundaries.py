"""Offline boundary experiment using real loopback HTTP, httpx, and OpenAI SDK.

Run from the repository root with .venv/bin/python and --output PATH.
No credentials, external requests, model generation, or tool execution occur.
This characterizes current behavior; it does not implement a timeout policy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import httpx
import openai

from daita.llm._lifecycle import NativeOwner, await_cleanup
from daita.llm.errors import ModelProviderError, interrupted_attempt_diagnostic
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelCallPolicy,
    ModelProfile,
    ModelRequest,
    ModelSensitivity,
    ModelStreamCompleted,
    TextBlock,
)
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from tests.support.stream_boundaries import HARD_SECONDS, MODEL, READ_SECONDS, endpoint


async def probe(layer, scenario, read_timeout):
    events: Counter[str] = Counter()
    result = {
        "layer": layer,
        "scenario": scenario,
        "read_seconds": read_timeout,
        "hard_seconds": HARD_SECONDS,
    }
    async with endpoint(scenario) as (url, requests, writes):
        timeout = httpx.Timeout(600, connect=5, read=read_timeout)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http:
            sdk = openai.AsyncOpenAI(
                api_key="offline-placeholder",
                base_url=url,
                http_client=http,
                max_retries=0,
                timeout=timeout,
            )
            provider = OpenAIResponsesProvider(
                MODEL, client=cast(Any, sdk), max_output_tokens=128
            )
            registration = ModelProviderRegistration(
                provider=provider,
                profile=ModelProfile(
                    id=provider.provider_id,
                    context_window_tokens=10000,
                    max_output_tokens=128,
                    supports_streaming=True,
                ),
                allowed_sensitivities=frozenset(
                    {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
                ),
            )
            router = ModelRouter(
                (registration,),
                retry_policy=RetryPolicy(
                    max_attempts_per_candidate=2, backoff_seconds=0
                ),
            )
            begun = time.monotonic()
            request = ModelRequest(
                messages=(
                    CanonicalMessage(
                        role=MessageRole.USER, content=(TextBlock("Reply x."),)
                    ),
                ),
                call_policy=ModelCallPolicy(
                    max_attempt_seconds=HARD_SECONDS,
                    first_progress_timeout_seconds=0.3,
                    progress_idle_timeout_seconds=0.2,
                    cleanup_timeout_seconds=0.2,
                    read_timeout_seconds=min(read_timeout, 120),
                ),
                max_total_tokens=1000,
                deadline=asyncio.get_running_loop().time() + HARD_SECONDS,
            )
            terminal = False
            try:
                # Raw clients need the harness deadline. Framework arms enforce
                # their own canonical deadline; the harness is only a later guard.
                async with asyncio.timeout(
                    HARD_SECONDS + (0.5 if layer in {"adapter", "router"} else 0)
                ):
                    if layer == "http":
                        async with http.stream(
                            "POST",
                            url + "/responses",
                            json={
                                "model": MODEL,
                                "input": "Reply x.",
                                "stream": True,
                                "max_output_tokens": 128,
                            },
                        ) as response:
                            async for chunk in response.aiter_bytes():
                                events["http_body_chunk"] += 1
                                terminal |= b'"response.completed"' in chunk
                    elif layer == "sdk":
                        stream = await sdk.responses.create(
                            model=MODEL,
                            input="Reply x.",
                            stream=True,
                            max_output_tokens=128,
                        )
                        async with stream:
                            async for event in stream:
                                events[event.type] += 1
                                terminal |= event.type == "response.completed"
                    else:
                        source = provider if layer == "adapter" else router
                        async for canonical_event in source.stream(request):
                            events[type(canonical_event).__name__] += 1
                            if isinstance(canonical_event, ModelStreamCompleted):
                                terminal = True
                                result["usage_complete"] = (
                                    canonical_event.response.usage.cost_estimate.status.value
                                    == "complete"
                                )
                result["outcome"] = "terminal" if terminal else "eof_without_terminal"
            except Exception as error:  # noqa: BLE001
                # Classify boundary failures for the diagnostic assertions.
                result["outcome"] = type(error).__name__
                if isinstance(error, ModelProviderError):
                    result["failure_code"] = error.code.value
                    result["usage_complete"] = (
                        error.usage.cost_estimate.status.value == "complete"
                    )
                    result["diagnostic_code"] = (
                        error.diagnostic.code if error.diagnostic else None
                    )
                diagnostic = interrupted_attempt_diagnostic(error)
                if (
                    diagnostic is None
                    and isinstance(error, ModelProviderError)
                    and error.diagnostic
                ):
                    diagnostic = error.diagnostic.attempt
                if diagnostic is not None:
                    from daita._json import thaw_json

                    result["attempt_diagnostic"] = thaw_json(diagnostic)
            finally:
                result["seconds"] = round(time.monotonic() - begun, 4)
                await router.close()
                await provider.close()
                await sdk.close()
            result.update(
                events=dict(events),
                terminal=terminal,
                paths=list(requests),
                server_writes=len(writes),
            )
    return result


async def cleanup_probe():
    started = time.monotonic()
    cleanup = asyncio.create_task(asyncio.sleep(0.3))
    try:
        await await_cleanup(cleanup, deadline=started + 0.08, owner=NativeOwner())
    except ModelProviderError as error:
        assert error.code.value == "cleanup_timeout"
    elapsed = time.monotonic() - started
    assert not cleanup.done() and elapsed < 0.2
    await cleanup
    return {
        "scenario": "bounded_delayed_cleanup",
        "cleanup_deadline_seconds": 0.08,
        "elapsed_seconds": round(elapsed, 4),
        "assertions": "passed",
    }


async def main(output):
    rows = []
    for scenario in (
        "silent",
        "startup_only",
        "comments",
        "empty_events",
        "text_trickle",
        "partial_tool",
        "eof_without_terminal",
        "complete",
    ):
        for layer in ("http", "sdk", "adapter", "router"):
            row = await probe(layer, scenario, 600)
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "layer",
                            "scenario",
                            "outcome",
                            "seconds",
                            "terminal",
                            "paths",
                        )
                    }
                ),
                flush=True,
            )
    for scenario in ("startup_only", "comments", "empty_events"):
        for layer in ("http", "sdk", "adapter", "router"):
            rows.append(await probe(layer, scenario, READ_SECONDS))
    for layer in ("adapter", "router"):
        rows.append(await probe(layer, "count_failure", 600))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "openai": openai.__version__,
                "httpx": httpx.__version__,
                "rows": rows,
                "assertions": "pending",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    # Assertions distinguish transport liveness, SDK parsing, canonical progress,
    # incomplete EOF, and safe router admission retries without model generation.
    for row in rows:
        scenario, layer = row["scenario"], row["layer"]
        generation_calls = sum(p.endswith("/responses") for p in row["paths"])
        assert generation_calls == (0 if scenario == "count_failure" else 1), row
        if scenario == "complete":
            assert row["terminal"], row
        elif scenario == "eof_without_terminal":
            assert not row["terminal"], row
            if layer in {"adapter", "router"}:
                assert row.get("failure_code") == "malformed_response", row
        elif scenario == "count_failure":
            assert len(row["paths"]) == (2 if layer == "router" else 1), row
            assert row["usage_complete"], row
        else:
            assert not row["terminal"], row
            if row["read_seconds"] == READ_SECONDS and scenario == "startup_only":
                assert 0.15 <= row["seconds"] < 0.55, row
            else:
                if layer in {"adapter", "router"} and scenario != "text_trickle":
                    assert 0.15 <= row["seconds"] < 0.65, row
                else:
                    assert 0.6 <= row["seconds"] < 2, row
        if scenario == "comments" and layer == "sdk":
            assert row["events"] == {"response.created": 1}, row
        if scenario == "empty_events" and layer in {"adapter", "router"}:
            assert not row["events"], row
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "openai": openai.__version__,
                "httpx": httpx.__version__,
                "rows": rows,
                "cleanup": await cleanup_probe(),
                "assertions": "passed",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"PASS: {len(rows)} boundary cases and one cleanup probe; {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.output))
