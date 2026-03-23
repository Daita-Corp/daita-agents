"""
Unit tests for EmailPlugin.

Tests comma-separated recipient parsing without a real SMTP/IMAP connection.
"""

import pytest
from daita.plugins.email import EmailPlugin

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
    plugin = EmailPlugin(
        email_address="test@x.com",
        password="p",
        imap_host="imap.x.com",
        smtp_host="smtp.x.com",
    )

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["to"] = to
        captured["cc"] = cc
        return {"message_id": "123"}

    plugin.send_email = fake_send

    await plugin._tool_send_email(
        {
            "to": "alice@x.com, bob@x.com",
            "subject": "Hello",
            "body": "Hi there",
        }
    )

    assert captured["to"] == ["alice@x.com", "bob@x.com"]


async def test_tool_send_email_splits_comma_separated_cc(monkeypatch):
    plugin = EmailPlugin(
        email_address="test@x.com",
        password="p",
        imap_host="imap.x.com",
        smtp_host="smtp.x.com",
    )

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["cc"] = cc
        return {"message_id": "456"}

    plugin.send_email = fake_send

    await plugin._tool_send_email(
        {
            "to": "alice@x.com",
            "subject": "Hello",
            "body": "Hi",
            "cc": "carol@x.com, dave@x.com",
        }
    )

    assert captured["cc"] == ["carol@x.com", "dave@x.com"]


async def test_tool_send_email_list_to_passthrough():
    plugin = EmailPlugin(
        email_address="test@x.com",
        password="p",
        imap_host="imap.x.com",
        smtp_host="smtp.x.com",
    )

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["to"] = to
        return {"message_id": "789"}

    plugin.send_email = fake_send

    await plugin._tool_send_email(
        {
            "to": ["alice@x.com", "bob@x.com"],
            "subject": "Test",
            "body": "Body",
        }
    )

    assert captured["to"] == ["alice@x.com", "bob@x.com"]


async def test_tool_send_email_empty_cc_becomes_none():
    plugin = EmailPlugin(
        email_address="test@x.com",
        password="p",
        imap_host="imap.x.com",
        smtp_host="smtp.x.com",
    )

    captured = {}

    async def fake_send(to, subject, body, html=False, cc=None, bcc=None):
        captured["cc"] = cc
        return {"message_id": "000"}

    plugin.send_email = fake_send

    await plugin._tool_send_email(
        {
            "to": "alice@x.com",
            "subject": "Test",
            "body": "Body",
            # no cc key
        }
    )

    assert captured["cc"] is None
