from __future__ import annotations

from collections.abc import Mapping
import json

import pytest

from daita.cli import CliRequestError, EXIT_ERROR, EXIT_OK, EXIT_USAGE, main


class RecordingRequest:
    def __init__(self, result: object | None = None) -> None:
        self.result = {"accepted": True} if result is None else result
        self.calls: list[tuple[str, dict[str, object], str | None]] = []

    async def __call__(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        idempotency_key: str | None,
    ) -> object:
        self.calls.append((method, dict(params), idempotency_key))
        return self.result


@pytest.mark.parametrize(
    ("argv", "method", "params", "key"),
    [
        (
            ["serve", "atlas", "--openai-model", "gpt-test"],
            "host.serve",
            {
                "openai_model": "gpt-test",
                "context_window_tokens": 128_000,
                "max_output_tokens": 4_096,
                "cadence_seconds": 1.0,
            },
            None,
        ),
        (
            ["agent", "init", "atlas", "--idempotency-key", "init-1"],
            "agent.init",
            {},
            "init-1",
        ),
        (
            ["agent", "create", "atlas", "--idempotency-key", "create-1"],
            "agent.create",
            {},
            "create-1",
        ),
        (["host", "status", "atlas"], "host.status", {}, None),
        (["host", "health", "atlas"], "host.health", {}, None),
        (
            [
                "source",
                "attach",
                "atlas",
                "sqlite",
                "/data/sales.db",
                "--idempotency-key",
                "source-1",
            ],
            "source.attach",
            {"kind": "sqlite", "path": "/data/sales.db"},
            "source-1",
        ),
        (
            [
                "source",
                "attach",
                "atlas",
                "sqlite",
                "/data/sales.db",
                "--write-access",
                "--idempotency-key",
                "source-write-1",
            ],
            "source.attach",
            {
                "kind": "sqlite",
                "path": "/data/sales.db",
                "write_access": True,
            },
            "source-write-1",
        ),
        (["model", "status", "atlas"], "model.status", {}, None),
        (["model", "show", "atlas"], "model.show", {}, None),
        (
            [
                "model",
                "set",
                "atlas",
                "ollama:qwen-test",
                "--allow-sensitivity",
                "public",
                "--idempotency-key",
                "model-1",
            ],
            "model.set",
            {
                "model_id": "ollama:qwen-test",
                "context_window_tokens": 128_000,
                "max_output_tokens": 4_096,
                "allowed_sensitivities": ["public"],
            },
            "model-1",
        ),
        (
            [
                "chat",
                "submit",
                "atlas",
                "show totals",
                "--session-id",
                "session-1",
                "--idempotency-key",
                "chat-1",
            ],
            "chat.submit",
            {"message": "show totals", "session_id": "session-1"},
            "chat-1",
        ),
        (
            ["operation", "inspect", "atlas", "operation-1"],
            "operation.inspect",
            {"operation_id": "operation-1"},
            None,
        ),
        (
            ["operation", "list", "atlas", "--status", "succeeded", "--limit", "5"],
            "operation.list",
            {"statuses": ["succeeded"], "limit": 5},
            None,
        ),
        (
            [
                "operation",
                "cancel",
                "atlas",
                "operation-1",
                "--reason",
                "no longer needed",
                "--idempotency-key",
                "cancel-1",
            ],
            "operation.cancel",
            {"operation_id": "operation-1", "reason": "no longer needed"},
            "cancel-1",
        ),
        (
            [
                "approval",
                "approve",
                "atlas",
                "approval-1",
                "--actor",
                "user-1",
                "--reason",
                "reviewed",
                "--idempotency-key",
                "decision-1",
            ],
            "approval.approve",
            {"approval_id": "approval-1", "actor_id": "user-1", "reason": "reviewed"},
            "decision-1",
        ),
        (
            [
                "approval",
                "reject",
                "atlas",
                "approval-2",
                "--actor",
                "user-1",
                "--reason",
                "scope is too broad",
                "--idempotency-key",
                "decision-2",
            ],
            "approval.reject",
            {
                "approval_id": "approval-2",
                "actor_id": "user-1",
                "reason": "scope is too broad",
            },
            "decision-2",
        ),
        (
            ["approval", "list", "atlas", "--status", "pending", "--limit", "5"],
            "approval.list",
            {"statuses": ["pending"], "limit": 5},
            None,
        ),
        (
            ["approval", "inspect", "atlas", "approval-1"],
            "approval.inspect",
            {"approval_id": "approval-1"},
            None,
        ),
        (
            ["source", "list", "atlas", "--include-detached"],
            "source.list",
            {"include_detached": True},
            None,
        ),
        (
            [
                "source",
                "detach",
                "atlas",
                "source-1",
                "--idempotency-key",
                "detach-1",
            ],
            "source.detach",
            {"source_id": "source-1"},
            "detach-1",
        ),
        (
            ["source", "health", "atlas", "source-1"],
            "source.health",
            {"source_id": "source-1"},
            None,
        ),
        (
            [
                "catalog",
                "search",
                "atlas",
                "customers",
                "--source-id",
                "source-1",
                "--kind",
                "table",
            ],
            "catalog.search",
            {
                "query": "customers",
                "source_ids": ["source-1"],
                "resource_kinds": ["table"],
                "limit": 20,
            },
            None,
        ),
        (
            ["catalog", "show", "atlas", "resource-1"],
            "catalog.show",
            {"resource_id": "resource-1"},
            None,
        ),
        (
            ["memory", "list", "atlas", "--source-id", "source-1", "--limit", "5"],
            "memory.list",
            {
                "source_id": "source-1",
                "include_superseded": False,
                "include_rejected": False,
                "limit": 5,
            },
            None,
        ),
        (
            ["memory", "inspect", "atlas", "memory-1"],
            "memory.inspect",
            {"memory_id": "memory-1"},
            None,
        ),
        (["skill", "list", "atlas"], "skill.list", {}, None),
        (
            ["skill", "inspect", "atlas", "skill-1"],
            "skill.inspect",
            {"skill_id": "skill-1"},
            None,
        ),
        (
            ["events", "read", "atlas", "--after", "7", "--limit", "25"],
            "events.read",
            {"after": 7, "limit": 25},
            None,
        ),
        (
            ["events", "follow", "atlas", "--after", "7"],
            "events.follow",
            {"after": 7, "limit": 100, "poll_seconds": 0.1},
            None,
        ),
        (
            [
                "monitor",
                "propose",
                "atlas",
                "daily-sales",
                "--definition",
                '{"condition":{"kind":"changed"},"schedule":{"interval_seconds":60}}',
                "--source-operation-id",
                "operation-1",
                "--idempotency-key",
                "proposal-1",
            ],
            "monitor.propose",
            {
                "monitor_id": "daily-sales",
                "definition": {
                    "condition": {"kind": "changed"},
                    "schedule": {"interval_seconds": 60},
                },
                "source_operation_id": "operation-1",
            },
            "proposal-1",
        ),
        (
            [
                "monitor",
                "propose-natural",
                "atlas",
                "daily-sales",
                "Every 5 minutes, Count sales for source source-1 when rows.0.total > 10",
                "--idempotency-key",
                "natural-1",
            ],
            "monitor.propose_natural",
            {
                "monitor_id": "daily-sales",
                "request": "Every 5 minutes, Count sales for source source-1 when rows.0.total > 10",
            },
            "natural-1",
        ),
        (
            [
                "monitor",
                "list",
                "atlas",
                "--status",
                "enabled",
                "--status",
                "paused",
                "--include-deleted",
                "--limit",
                "20",
            ],
            "monitor.list",
            {
                "statuses": ["enabled", "paused"],
                "include_deleted": True,
                "limit": 20,
            },
            None,
        ),
        (
            ["monitor", "inspect", "atlas", "daily-sales"],
            "monitor.inspect",
            {"monitor_id": "daily-sales"},
            None,
        ),
        (
            [
                "monitor",
                "confirm",
                "atlas",
                "proposal-1",
                "--candidate-hash",
                "abc123",
                "--actor",
                "user-1",
                "--reason",
                "confirmed",
                "--idempotency-key",
                "confirmation-1",
            ],
            "monitor.confirm",
            {
                "proposal_id": "proposal-1",
                "candidate_hash": "abc123",
                "actor_id": "user-1",
                "reason": "confirmed",
            },
            "confirmation-1",
        ),
        (
            [
                "monitor",
                "pause",
                "atlas",
                "daily-sales",
                "--actor",
                "user-1",
                "--reason",
                "maintenance",
                "--operation-id",
                "operation-2",
                "--idempotency-key",
                "pause-1",
            ],
            "monitor.pause",
            {
                "monitor_id": "daily-sales",
                "actor_id": "user-1",
                "reason": "maintenance",
                "operation_id": "operation-2",
            },
            "pause-1",
        ),
        (
            [
                "monitor",
                "resume",
                "atlas",
                "daily-sales",
                "--actor",
                "user-1",
                "--reason",
                "maintenance complete",
                "--idempotency-key",
                "resume-1",
            ],
            "monitor.resume",
            {
                "monitor_id": "daily-sales",
                "actor_id": "user-1",
                "reason": "maintenance complete",
            },
            "resume-1",
        ),
        (
            [
                "monitor",
                "run-now",
                "atlas",
                "daily-sales",
                "--lease-seconds",
                "15",
                "--idempotency-key",
                "run-now-1",
            ],
            "monitor.run_now",
            {"monitor_id": "daily-sales", "lease_seconds": 15.0},
            "run-now-1",
        ),
        (
            [
                "monitor",
                "delete",
                "atlas",
                "daily-sales",
                "--actor",
                "user-1",
                "--reason",
                "retired",
                "--idempotency-key",
                "delete-1",
            ],
            "monitor.delete",
            {"monitor_id": "daily-sales", "actor_id": "user-1", "reason": "retired"},
            "delete-1",
        ),
    ],
)
def test_commands_dispatch_one_thin_request(
    argv: list[str],
    method: str,
    params: dict[str, object],
    key: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = RecordingRequest()

    assert main(argv, request=request) == EXIT_OK

    assert request.calls == [(method, params, key)]
    assert json.loads(capsys.readouterr().out) == {"accepted": True}


def test_mutation_requires_idempotency_key_and_does_not_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = RecordingRequest()

    assert main(["chat", "submit", "atlas", "hello"], request=request) == EXIT_USAGE

    assert request.calls == []
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "usage_error"
    assert "--idempotency-key" in error["error"]["message"]


def test_monitor_definition_must_be_json_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = RecordingRequest()

    assert (
        main(
            [
                "monitor",
                "propose",
                "atlas",
                "daily-sales",
                "--definition",
                "[]",
                "--idempotency-key",
                "proposal-1",
            ],
            request=request,
        )
        == EXIT_USAGE
    )

    assert request.calls == []
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "usage_error"


def test_request_error_is_bounded_and_returns_stable_failure_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingRequest:
        async def __call__(
            self,
            method: str,
            params: Mapping[str, object],
            *,
            idempotency_key: str | None,
        ) -> object:
            raise CliRequestError("approval_conflict", "approval was already decided")

    assert (
        main(["operation", "inspect", "atlas", "op-1"], request=FailingRequest())
        == EXIT_ERROR
    )

    assert json.loads(capsys.readouterr().err) == {
        "error": {
            "code": "approval_conflict",
            "message": "approval was already decided",
        }
    }


def test_follow_writes_each_stream_item_as_one_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class StreamingRequest:
        async def __call__(
            self,
            method: str,
            params: Mapping[str, object],
            *,
            idempotency_key: str | None,
        ) -> object:
            async def items():
                yield {"sequence": 1, "type": "operation.started"}
                yield {"sequence": 2, "type": "operation.completed"}

            return items()

    assert main(["events", "follow", "atlas"], request=StreamingRequest()) == EXIT_OK

    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"sequence": 1, "type": "operation.started"},
        {"sequence": 2, "type": "operation.completed"},
    ]


def test_root_is_cli_routing_context_not_an_rpc_parameter(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = RecordingRequest()

    assert (
        main(
            ["--root", str(tmp_path), "host", "status", "atlas"],
            request=request,
        )
        == EXIT_OK
    )

    assert request.calls == [("host.status", {}, None)]
    assert json.loads(capsys.readouterr().out) == {"accepted": True}


def test_default_transport_resolves_the_selected_agent_home(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from daita.hosting import local_protocol

    homes = []
    envelopes = []

    class FakeClient:
        def __init__(self, agent_home) -> None:
            homes.append(agent_home)

        async def request(self, envelope):
            envelopes.append(envelope)
            return local_protocol.LocalSuccessResponse.create(
                request_id=envelope.request_id,
                result={"state": "running"},
            )

    monkeypatch.setattr(local_protocol, "LocalAgentClient", FakeClient)

    assert main(["--root", str(tmp_path), "host", "status", "atlas"]) == EXIT_OK

    assert homes == [tmp_path / "agents" / "atlas"]
    assert len(envelopes) == 1
    assert envelopes[0].method == "host.status"
    assert envelopes[0].params.to_dict() == {}
    assert json.loads(capsys.readouterr().out) == {"state": "running"}


@pytest.mark.parametrize(
    "definition",
    (
        '{"schedule":{"interval_seconds":NaN}}',
        '{"schedule":{},"schedule":{}}',
    ),
)
def test_monitor_definition_rejects_non_strict_json(
    definition: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "monitor",
                "propose",
                "atlas",
                "daily-sales",
                "--definition",
                definition,
                "--idempotency-key",
                "proposal-1",
            ],
            request=RecordingRequest(),
        )
        == EXIT_USAGE
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "usage_error"


def test_default_agent_init_bootstraps_without_a_socket_and_is_replay_safe(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = [
        "--root",
        str(tmp_path),
        "agent",
        "init",
        "atlas",
        "--idempotency-key",
        "init-1",
    ]

    assert main(command) == EXIT_OK
    created = json.loads(capsys.readouterr().out)
    assert created["created"] is True
    assert created["home"] == str(tmp_path / "agents" / "atlas")

    assert main(command) == EXIT_OK
    replay = json.loads(capsys.readouterr().out)
    assert replay == {**created, "created": False}


def test_default_first_run_configures_model_and_source_before_serve(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    create = [
        "--root",
        str(tmp_path),
        "agent",
        "create",
        "atlas",
        "--idempotency-key",
        "create-1",
    ]
    assert main(create) == EXIT_OK
    capsys.readouterr()

    source_root = tmp_path / "exports"
    source_root.mkdir()
    (source_root / "customers.csv").write_text("id\n1\n", encoding="utf-8")
    add_source = [
        "--root",
        str(tmp_path),
        "source",
        "add",
        "atlas",
        "local_files",
        str(source_root),
        "--idempotency-key",
        "source-1",
    ]
    assert main(add_source) == EXIT_OK
    attached = json.loads(capsys.readouterr().out)
    assert attached["active"] is True

    set_model = [
        "--root",
        str(tmp_path),
        "model",
        "set",
        "atlas",
        "ollama:qwen-test",
        "--idempotency-key",
        "model-1",
    ]
    assert main(set_model) == EXIT_OK
    configured = json.loads(capsys.readouterr().out)
    assert configured["changed"] is True
    assert configured["revision"] == 1

    assert main(set_model) == EXIT_OK
    replay = json.loads(capsys.readouterr().out)
    assert replay == {**configured, "changed": False}

    assert main(["--root", str(tmp_path), "model", "show", "atlas"]) == EXIT_OK
    shown = json.loads(capsys.readouterr().out)
    assert shown["route"]["revision"] == 1
    assert shown["route"]["candidates"][0]["provider_id"] == "ollama:qwen-test"
    assert shown["route"]["candidates"][0]["secret_reference"] is None

    from daita.hosting import local_server

    class OneShotServer:
        def __init__(self, host) -> None:
            self.host = host
            self.socket_path = host.home / "run" / "agent.sock"

        async def start(self) -> None:
            await self.host.start()

        async def serve_forever(self) -> None:
            return None

        async def stop(self, *, drain: bool) -> None:
            await self.host.stop(drain=drain)

    monkeypatch.setattr(local_server, "LocalAgentServer", OneShotServer)
    assert main(["--root", str(tmp_path), "serve", "atlas"]) == EXIT_OK
    served = json.loads(capsys.readouterr().out)
    assert served["model_profile_id"] == "ollama:qwen-test"
    assert served["state"] == "running"


def test_chat_agent_shorthand_selects_line_oriented_interactive_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = RecordingRequest()

    assert main(["chat", "atlas"], request=request) == EXIT_OK

    assert request.calls == [
        ("chat.interactive", {"poll_seconds": 0.1}, None),
    ]
    assert json.loads(capsys.readouterr().out) == {"accepted": True}


def test_serve_uses_foreground_host_and_local_server(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from daita.hosting.host import AgentHost
    from daita.hosting import local_server

    observed: dict[str, object] = {}

    class FakeHost:
        id = "agent-atlas"
        model_profile = type("Profile", (), {"id": "openai:gpt-test"})()

    async def open_host(cls, name, **kwargs):
        observed["name"] = name
        observed["open"] = kwargs
        return FakeHost()

    class FakeServer:
        socket_path = tmp_path / "agents" / "atlas" / "run" / "agent.sock"

        def __init__(self, host) -> None:
            observed["host"] = host

        async def start(self) -> None:
            observed["started"] = True

        async def serve_forever(self) -> None:
            observed["served"] = True

        async def stop(self, *, drain: bool) -> None:
            observed["stopped"] = drain

    monkeypatch.setattr(AgentHost, "open", classmethod(open_host))
    monkeypatch.setattr(local_server, "LocalAgentServer", FakeServer)

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "serve",
                "atlas",
                "--openai-model",
                "gpt-test",
            ]
        )
        == EXIT_OK
    )

    assert observed["name"] == "atlas"
    assert observed["started"] is True
    assert observed["served"] is True
    assert observed["stopped"] is True
    output = json.loads(capsys.readouterr().out)
    assert output["agent_id"] == "agent-atlas"
    assert output["model_profile_id"] == "openai:gpt-test"

    observed.clear()
    assert main(["--root", str(tmp_path), "serve", "atlas"]) == EXIT_OK
    open_arguments = observed["open"]
    assert isinstance(open_arguments, dict)
    assert open_arguments["model"] is None
    assert open_arguments["model_profile"] is None
    assert observed["served"] is True
    capsys.readouterr()
