"""Component-owned tests split from ``test_public_surfaces.py``."""

from __future__ import annotations

from tests.artifacts._public_surface_support import (
    ArtifactDeliveryReceipt,
    MockModelProvider,
    Path,
    RunInput,
    Transcript,
    _profile,
    _surface_records,
    cli,
    pytest,
)


async def test_cli_run_json_contains_refs_and_receipts_but_no_payload_or_grant_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, receipt, result = _surface_records()

    class _FakeAgent:
        async def run(self, message: str, *, conversation_id: str | None = None):
            del message, conversation_id
            return result

        async def transcript(self, run_id: str):
            return Transcript(
                RunInput(
                    id=run_id,
                    agent_id="agent",
                    message="create file",
                    conversation_id=result.conversation_id,
                    created_at=result.created_at,
                )
            )

        async def close(self) -> None:
            return None

    async def fake_open(*args, **kwargs):
        del args, kwargs
        return _FakeAgent()

    provider = MockModelProvider((), provider_id="mock:cli-artifacts")
    monkeypatch.setattr(cli.Agent, "open", staticmethod(fake_open))
    monkeypatch.setattr(
        cli,
        "_model_configuration",
        lambda *args, **kwargs: (provider, _profile(provider)),
    )
    args = cli.build_parser().parse_args(
        ["--root", str(tmp_path), "run", "agent", "create file", "--model", "x:y"]
    )
    mapping = await cli._execute(args)
    assert isinstance(mapping, dict)
    assert mapping["artifacts"][0]["artifact_id"] == ref.artifact_id
    assert mapping["artifact_deliveries"][0]["saved_path"] == receipt.saved_path
    rendered = str(mapping)
    assert "content" not in rendered
    assert "grant_digest" not in rendered
    assert "destination root" not in rendered


async def test_cli_artifact_save_uses_direct_destination_once_and_returns_structured_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, receipt, _ = _surface_records()
    destination = tmp_path / "direct"
    destination.mkdir()
    calls: list[tuple[str, Path | None, str | None]] = []

    class _FakeAgent:
        async def save_artifact(
            self,
            artifact_id: str,
            selected: Path | None,
            *,
            filename: str | None,
        ) -> ArtifactDeliveryReceipt:
            calls.append((artifact_id, selected, filename))
            return receipt

        async def close(self) -> None:
            return None

    async def fake_open(*args, **kwargs):
        del args, kwargs
        return _FakeAgent()

    monkeypatch.setattr(cli.Agent, "open", staticmethod(fake_open))
    args = cli.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "artifacts",
            "save",
            "agent",
            ref.artifact_id,
            "--destination",
            str(destination),
            "--filename",
            "renamed.txt",
        ]
    )
    mapping = await cli._execute(args)
    assert calls == [(ref.artifact_id, destination, "renamed.txt")]
    assert isinstance(mapping, dict)
    assert mapping["saved_path"] == receipt.saved_path
    assert "content" not in mapping
