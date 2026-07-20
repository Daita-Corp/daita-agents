"""Run a production-shaped Daita v2 foreground host on a private Unix socket."""

from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
from pathlib import Path
import signal

from daita import AgentHost, SQLiteSource, create_llm_provider
from daita.hosting import LocalAgentServer
from daita.llm import ModelProfile


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dedicated explicit v2 state root (never the v1 ~/.daita root).",
    )
    parser.add_argument("--agent", default="data-team")
    parser.add_argument("--model", default="openai:gpt-4.1-mini")
    parser.add_argument("--context-window-tokens", type=int, default=128_000)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--cadence-seconds", type=float, default=1.0)
    parser.add_argument(
        "--sqlite",
        type=Path,
        help="Optional absolute read-only SQLite source to attach on creation.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new durable identity instead of opening an existing one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the non-secret configuration without touching state or SDKs.",
    )
    return parser.parse_args()


async def serve(config: argparse.Namespace) -> None:
    root = config.root.expanduser().resolve(strict=False)
    if config.sqlite is not None:
        sqlite_path = config.sqlite.expanduser().resolve(strict=True)
        if not sqlite_path.is_absolute():
            raise ValueError("--sqlite must resolve to an absolute path")
    else:
        sqlite_path = None
    if sqlite_path is not None and not config.create:
        raise ValueError("--sqlite is accepted only with --create")
    if config.dry_run:
        print(f"agent: {config.agent}")
        print(f"root: {root}")
        print(f"model: {config.model}")
        print(f"mode: {'create' if config.create else 'open'}")
        print(f"read-only SQLite: {sqlite_path or 'none'}")
        return

    provider = create_llm_provider(
        config.model,
        max_output_tokens=config.max_output_tokens,
    )
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=config.context_window_tokens,
        max_output_tokens=config.max_output_tokens,
        supports_tools=True,
        supports_streaming=True,
    )
    factory = AgentHost.create if config.create else AgentHost.open
    host = await factory(
        config.agent,
        root=root,
        model=provider,
        model_profile=profile,
        cadence_seconds=config.cadence_seconds,
    )
    server = LocalAgentServer(host)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(name, stopping.set)
        except NotImplementedError:
            pass
    try:
        await server.start()
        if sqlite_path is not None:
            source = await host.attach(
                SQLiteSource(sqlite_path, name=sqlite_path.stem),
                idempotency_key=(
                    "deployment-source:"
                    + sha256(str(sqlite_path).encode("utf-8")).hexdigest()
                ),
            )
            print(f"attached read-only source: {source.id}")
        print(f"agent: {host.name} ({host.id})")
        print(f"state: {host.home}")
        print(f"socket: {server.socket_path}")
        await stopping.wait()
    finally:
        await server.stop(drain=True)


if __name__ == "__main__":
    asyncio.run(serve(arguments()))
