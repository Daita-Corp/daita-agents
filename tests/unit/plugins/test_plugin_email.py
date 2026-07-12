"""
Unit tests for EmailPlugin.

Tests comma-separated recipient parsing without a real SMTP/IMAP connection.
"""

import pytest
from daita.plugins.manifest import PluginKind
from daita.plugins.registry import ExtensionRegistry
from daita.plugins.email import EmailPlugin
from daita.runtime import Operation, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin():
    return EmailPlugin(
        email_address="test@x.com",
        password="p",
        imap_host="imap.x.com",
        smtp_host="smtp.x.com",
    )


def test_email_connection_access_requires_connect():
    plugin = make_plugin()

    with pytest.raises(RuntimeError, match="IMAP connection"):
        _ = plugin.imap
    with pytest.raises(RuntimeError, match="SMTP connection"):
        _ = plugin.smtp


async def test_email_connection_access_tracks_lifecycle(monkeypatch):
    import imaplib
    import smtplib
    from unittest.mock import MagicMock

    imap = MagicMock()
    smtp = MagicMock()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", lambda *args: imap)
    monkeypatch.setattr(smtplib, "SMTP", lambda *args: smtp)
    plugin = make_plugin()

    await plugin.connect()
    assert plugin.imap is imap
    assert plugin.smtp is smtp

    await plugin.disconnect()
    imap.logout.assert_called_once()
    smtp.quit.assert_called_once()
    with pytest.raises(RuntimeError, match="IMAP connection"):
        _ = plugin.imap


async def test_list_emails_skips_malformed_fetch_result():
    from unittest.mock import MagicMock

    plugin = make_plugin()
    imap = MagicMock()
    imap.search.return_value = ("OK", [b"1"])
    imap.fetch.return_value = ("OK", [b"malformed"])
    plugin._imap = imap
    plugin._smtp = MagicMock()

    assert await plugin.list_emails() == []


def test_email_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "email"
    assert plugin.manifest.kind is PluginKind.CONNECTOR
    assert registry.plugin_ids == ("email",)
    assert {capability.id for capability in registry.capabilities} == {
        "email.message.list",
        "email.message.read",
        "email.message.send",
        "email.message.reply",
        "email.message.search",
    }
    assert {view.name for view in registry.tool_views} == {
        "list_emails",
        "read_email",
        "send_email",
        "reply_to_email",
        "search_emails",
    }
    assert registry.evidence_schemas[0].kind == "email.operation.result"


def test_email_projected_tools_carry_declared_capability_metadata():
    plugin = make_plugin()

    by_name = projected_tools(plugin)

    assert by_name["list_emails"].capability_ids == ("email.message.list",)
    assert by_name["list_emails"].side_effecting is False
    assert by_name["list_emails"].idempotent is True
    assert by_name["send_email"].capability_ids == ("email.message.send",)
    assert by_name["send_email"].side_effecting is True
    assert by_name["read_email"].capability_ids == ("email.message.read",)
    assert by_name["read_email"].side_effecting is True


async def test_email_executor_returns_typed_operation_evidence(monkeypatch):
    plugin = make_plugin()

    async def fake_list_emails(folder="INBOX", limit=10, unread_only=False, **kwargs):
        assert folder == "INBOX"
        assert limit == 2
        assert unread_only is True
        return [{"id": "1", "subject": "Hello"}]

    monkeypatch.setattr(plugin, "list_emails", fake_list_emails)
    registry = ExtensionRegistry()
    registry.register(plugin)

    executor = registry.get_executor("email.message")
    operation = Operation(id="op-1", operation_type="email.message.list")
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="email.message.list",
        executor_id="email.message",
        input={"folder": "INBOX", "limit": 2, "unread_only": True},
        required_evidence=frozenset({"email.operation.result"}),
    )

    evidence = await executor.execute(
        task,
        operation,
        {"tool_view": {"name": "list_emails"}},
    )

    assert len(evidence) == 1
    assert evidence[0].kind == "email.operation.result"
    assert evidence[0].owner == "email"
    assert evidence[0].payload["operation"] == "list_emails"
    assert evidence[0].payload["request"]["limit"] == 2
    assert evidence[0].payload["result"] == {
        "emails": [{"id": "1", "subject": "Hello"}],
        "count": 1,
    }


# ---------------------------------------------------------------------------
# _parse_recipients (static helper)
# ---------------------------------------------------------------------------


def test_parse_single_address():
    result = EmailPlugin._parse_recipients("alice@example.com")
    assert result == ["alice@example.com"]


def test_parse_comma_separated_addresses():
    result = EmailPlugin._parse_recipients("alice@example.com, bob@example.com")
    assert result == ["alice@example.com", "bob@example.com"]


def test_parse_comma_no_spaces():
    result = EmailPlugin._parse_recipients("alice@x.com,bob@x.com,carol@x.com")
    assert result == ["alice@x.com", "bob@x.com", "carol@x.com"]


def test_parse_strips_whitespace():
    result = EmailPlugin._parse_recipients("  alice@x.com  ,  bob@x.com  ")
    assert result == ["alice@x.com", "bob@x.com"]


def test_parse_list_passthrough():
    addresses = ["alice@x.com", "bob@x.com"]
    result = EmailPlugin._parse_recipients(addresses)
    assert result == addresses


def test_parse_none_returns_empty():
    result = EmailPlugin._parse_recipients(None)
    assert result == []


def test_parse_empty_string_returns_empty():
    result = EmailPlugin._parse_recipients("")
    assert result == []


def test_parse_empty_list_returns_empty():
    result = EmailPlugin._parse_recipients([])
    assert result == []


# ---------------------------------------------------------------------------
# _tool_send_email — verifies comma split is applied before calling send_email
# ---------------------------------------------------------------------------


async def test_tool_send_email_splits_comma_separated_to(monkeypatch):
    plugin = make_plugin()

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["to"] = to
        captured["cc"] = cc
        return {"message_id": "123"}

    monkeypatch.setattr(plugin, "send_email", fake_send)

    await plugin._tool_send_email(
        {
            "to": "alice@x.com, bob@x.com",
            "subject": "Hello",
            "body": "Hi there",
        }
    )

    assert captured["to"] == ["alice@x.com", "bob@x.com"]


async def test_tool_send_email_splits_comma_separated_cc(monkeypatch):
    plugin = make_plugin()

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["cc"] = cc
        return {"message_id": "456"}

    monkeypatch.setattr(plugin, "send_email", fake_send)

    await plugin._tool_send_email(
        {
            "to": "alice@x.com",
            "subject": "Hello",
            "body": "Hi",
            "cc": "carol@x.com, dave@x.com",
        }
    )

    assert captured["cc"] == ["carol@x.com", "dave@x.com"]


async def test_tool_send_email_list_to_passthrough(monkeypatch):
    plugin = make_plugin()

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["to"] = to
        return {"message_id": "789"}

    monkeypatch.setattr(plugin, "send_email", fake_send)

    await plugin._tool_send_email(
        {
            "to": ["alice@x.com", "bob@x.com"],
            "subject": "Test",
            "body": "Body",
        }
    )

    assert captured["to"] == ["alice@x.com", "bob@x.com"]


async def test_tool_send_email_empty_cc_becomes_none(monkeypatch):
    plugin = make_plugin()

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["cc"] = cc
        return {"message_id": "000"}

    monkeypatch.setattr(plugin, "send_email", fake_send)

    await plugin._tool_send_email(
        {
            "to": "alice@x.com",
            "subject": "Test",
            "body": "Body",
            # no cc key
        }
    )

    assert captured["cc"] is None
