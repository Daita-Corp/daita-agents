"""Every surface member has a route; handlers only decode, call and encode."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import pytest

from daita.adapters.models import SourceRegistration
from daita.artifacts.models import ArtifactDestination, ArtifactDestinationKind
from daita.host.protocol import CWD_HEADER, REMOTE_AGENT_SURFACE
from daita.host.server import AREAS
from daita.hosting.sources import SourceEditResult
from daita.llm import subscription_auth
from daita.llm.models import ModelSensitivity
from daita.llm.subscription_auth import CodexOAuthCredential
from tests.host.conftest import StartHost
from tests.llm._subscription_support import _jwt
from tests.support.host import error_code, pending_interaction, value

pytestmark = pytest.mark.integration


@pytest.mark.unit
def test_every_surface_member_is_reached_by_exactly_the_listed_routes() -> None:
    reached = set().union(*(area.MEMBERS for area in AREAS))
    routes = [
        (method, route.path)
        for area in AREAS
        for route in area.ROUTES
        for method in route.methods or ()
        if method != "HEAD"
    ]

    assert reached == REMOTE_AGENT_SURFACE
    assert len(routes) == len(set(routes))


async def test_reads_take_json_query_values_and_writes_take_json_bodies(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()

    written = await harness.http.put(
        "/v1/agents/sales/memory",
        json={
            "text": "Revenue is paid subtotal.",
            "sensitivity": {"$type": "daita.ModelSensitivity", "v": "internal"},
        },
    )
    assert value(written) is None
    assert value(await harness.http.get("/v1/agents/sales/memory")) == (
        "Revenue is paid subtotal."
    )
    skills = await harness.http.get(
        "/v1/agents/sales/semantics", params={"source_id": json.dumps("s-unknown")}
    )
    assert value(skills) == ()

    unknown = await harness.http.put("/v1/agents/sales/memory", json={"txt": "x"})
    assert (unknown.status_code, error_code(unknown)) == (400, "invalid_value")
    wrong_type = await harness.http.put(
        "/v1/agents/sales/memory", json={"text": "x", "sensitivity": "internal"}
    )
    assert error_code(wrong_type) == "invalid_type"
    assert ModelSensitivity.INTERNAL.value == "internal"


async def test_relative_client_paths_resolve_against_the_request_folder_only(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()
    folder = harness.work()
    (folder / "exports").mkdir()
    database = folder / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    cwd = {CWD_HEADER: quote(str(folder))}

    destination = value(
        await harness.http.put(
            "/v1/agents/sales/export-destination",
            json={"directory": {"$type": "path", "v": "exports"}},
            headers=cwd,
        )
    )
    source = value(
        await harness.http.post(
            "/v1/agents/sales/sources/sqlite",
            json={"path": "orders.sqlite"},
            headers=cwd,
        )
    )
    without_folder = await harness.http.post(
        "/v1/agents/sales/sources/sqlite", json={"path": "orders.sqlite"}
    )

    assert isinstance(destination, ArtifactDestination)
    assert destination.kind is ArtifactDestinationKind.LOCAL_DIRECTORY
    assert isinstance(source, SourceRegistration)
    assert Path(str(source.configuration["path"])) == database.resolve()
    assert value(await harness.http.get("/v1/agents/sales/sources")) == (source,)
    assert error_code(without_folder) == "invalid_value"
    assert CWD_HEADER in without_folder.json()["error"]["message"]


async def test_a_source_edit_waits_for_its_confirmation_interaction(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    await harness.create_agent()
    folder = harness.work()
    for name in ("orders.sqlite", "orders-copy.sqlite"):
        with sqlite3.connect(folder / name) as connection:
            connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    source = value(
        await harness.http.post(
            "/v1/agents/sales/sources/sqlite",
            json={"path": str(folder / "orders.sqlite")},
        )
    )

    edit = asyncio.create_task(
        harness.http.post(
            f"/v1/agents/sales/sources/{source.id}/sqlite",
            json={
                "path": str(folder / "orders-copy.sqlite"),
                "confirmation_handler": True,
            },
        )
    )
    interaction = await pending_interaction(harness.http, "sales")
    assert interaction["kind"] == "source_edit_confirmation"
    assert interaction["request"]["preview"].current_source_id == source.id
    await harness.http.post(
        f"/v1/agents/sales/interactions/{interaction['id']}", json={"accept": True}
    )
    edited = value(await asyncio.wait_for(edit, 10))

    assert isinstance(edited, SourceEditResult)
    assert (
        Path(str(edited.source.configuration["path"]))
        == (folder / "orders-copy.sqlite").resolve()
    )


def _fake_openai_device_login(
    monkeypatch: pytest.MonkeyPatch, authorized: asyncio.Event
) -> None:
    """Fake only OpenAI's endpoints; the token poll waits for ``authorized``."""

    async def post_json(url: str, _body: object) -> subscription_auth._HttpResult:
        if url.endswith("/usercode"):
            payload: dict[str, object] = {
                "device_auth_id": "device-1",
                "user_code": "ABCD-EFGH",
                "interval": 1,
            }
        else:
            await authorized.wait()
            payload = {"authorization_code": "code-1", "code_verifier": "verifier-1"}
        return subscription_auth._HttpResult(200, json.dumps(payload).encode())

    async def post_form(_url: str, _body: object) -> subscription_auth._HttpResult:
        tokens = {
            "access_token": _jwt(),
            "refresh_token": "refresh-1",
            "expires_in": 3_600,
        }
        return subscription_auth._HttpResult(200, json.dumps(tokens).encode())

    monkeypatch.setattr(subscription_auth, "_post_json", post_json)
    monkeypatch.setattr(subscription_auth, "_post_form", post_form)


async def test_subscription_login_shows_its_prompt_as_an_interaction(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorized = asyncio.Event()
    _fake_openai_device_login(monkeypatch, authorized)
    harness = await start_host()
    await harness.create_agent()

    async with harness.events("sales") as events:
        login = asyncio.create_task(
            harness.http.post(
                "/v1/agents/sales/model/subscription-login", json={"provider": "codex"}
            )
        )
        _seq, _event, opened = await events.next(
            "interaction", lambda data: data["op"] == "opened"
        )
        _seq, _event, waiting = await events.next(
            "interaction",
            lambda data: data["interaction"]["request"]["progress"]
            == "Waiting for ChatGPT authorization",
        )
        authorized.set()
        credential = value(await asyncio.wait_for(login, 10))
        _seq, _event, closed = await events.next(
            "interaction", lambda data: data["op"] == "resolved"
        )

    prompt = opened["interaction"]["request"]["prompt"]
    assert opened["interaction"]["kind"] == "model_login"
    assert (prompt.verification_url, prompt.user_code) == (
        "https://auth.openai.com/codex/device",
        "ABCD-EFGH",
    )
    assert waiting["op"] == "updated"
    assert closed["interaction"]["state"] == "completed"
    assert CodexOAuthCredential.from_secret(credential).refresh_token == "refresh-1"
    assert value(await harness.http.get("/v1/agents/sales/interactions")) == ()


async def test_dismissing_a_subscription_login_cancels_it(
    start_host: StartHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_openai_device_login(monkeypatch, asyncio.Event())
    harness = await start_host()
    await harness.create_agent()
    login = asyncio.create_task(
        harness.http.post(
            "/v1/agents/sales/model/subscription-login", json={"provider": "codex"}
        )
    )
    interaction = await pending_interaction(harness.http, "sales")

    answer = await harness.http.post(
        f"/v1/agents/sales/interactions/{interaction['id']}",
        json={"decision": "approve"},
    )
    dismissed = await harness.http.post(
        f"/v1/agents/sales/interactions/{interaction['id']}/dismiss"
    )
    response = await asyncio.wait_for(login, 10)

    assert error_code(answer) == "invalid_value"
    assert value(dismissed)["state"] == "dismissed"
    assert (response.status_code, error_code(response)) == (409, "already_resolved")
    assert value(await harness.http.get("/v1/agents/sales/interactions")) == ()


async def test_every_failure_is_one_error_envelope(start_host: StartHost) -> None:
    harness = await start_host()

    missing_route = await harness.http.get("/v1/nowhere")
    wrong_method = await harness.http.patch("/v1/agents")
    missing_agent = await harness.http.get("/v1/agents/nobody/memory")
    bad_name = await harness.http.get("/v1/agents/-bad/memory")
    old_client = await harness.http.get(
        "/v1/agents", headers={"X-Daita-Version": "0.0.1"}
    )

    for response, status, code in (
        (missing_route, 404, "key_not_found"),
        (wrong_method, 405, "invalid_value"),
        (missing_agent, 404, "agent_not_found"),
        (bad_name, 400, "agent_name_invalid"),
        (old_client, 409, "version_mismatch"),
    ):
        assert response.status_code == status, response.text
        assert set(response.json()) == {"error"}
        assert error_code(response) == code
    attributes = old_client.json()["error"]["details"]["attributes"]
    assert attributes["protocol"] == 1
