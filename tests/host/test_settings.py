"""Host settings in host.toml: defaults, one-key writes that keep the rest."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from daita.host.paths import HostPaths
from daita.host.settings import read_setting, write_setting

pytestmark = pytest.mark.unit


def test_missing_settings_read_as_defaults(tmp_path: Path) -> None:
    paths = HostPaths(tmp_path)

    assert read_setting(paths, "notifications") is True
    assert read_setting(paths, "background_offer") is True
    assert not paths.settings.exists()


def test_writing_one_key_keeps_comments_other_keys_and_tables(
    tmp_path: Path,
) -> None:
    paths = HostPaths(tmp_path)
    original = (
        "# my Daita settings\n"
        "notifications = true # keep these\n"
        'theme = "dark"\n'
        "\n"
        "[extra]\n"
        "background_offer = true\n"
    )
    paths.settings.write_text(original, encoding="utf-8")

    write_setting(paths, "background_offer", False)
    write_setting(paths, "notifications", False)

    text = paths.settings.read_text(encoding="utf-8")
    assert text.startswith("# my Daita settings\nnotifications = false\n")
    assert tomllib.loads(text) == {
        "notifications": False,
        "theme": "dark",
        "background_offer": False,
        "extra": {"background_offer": True},
    }
    assert paths.settings.stat().st_mode & 0o777 == 0o600
    assert read_setting(paths, "background_offer") is False


def test_a_malformed_file_is_reported_and_never_rewritten(tmp_path: Path) -> None:
    paths = HostPaths(tmp_path)
    paths.settings.write_text("notifications = maybe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid TOML"):
        read_setting(paths, "notifications")
    with pytest.raises(ValueError, match="not valid TOML"):
        write_setting(paths, "background_offer", False)
    assert paths.settings.read_text(encoding="utf-8") == "notifications = maybe\n"


def test_a_setting_must_be_a_boolean(tmp_path: Path) -> None:
    paths = HostPaths(tmp_path)
    paths.settings.write_text('notifications = "off"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="true or false"):
        read_setting(paths, "notifications")


def test_a_line_edit_that_would_change_another_value_is_refused(
    tmp_path: Path,
) -> None:
    paths = HostPaths(tmp_path)
    # The key's text appears inside a multi-line string before any real key.
    original = 'note = """\nbackground_offer = true\n"""\n'
    paths.settings.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="by hand"):
        write_setting(paths, "background_offer", False)
    assert paths.settings.read_text(encoding="utf-8") == original
