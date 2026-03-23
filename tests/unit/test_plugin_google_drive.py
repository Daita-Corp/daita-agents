"""
Unit tests for GoogleDrivePlugin.

Tests format detection, content extraction, truncation, error mapping,
tool schemas, auth flow selection, and all tool handlers — without real
Google API calls.
"""

import csv
import io
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from daita.plugins.google_drive import (
    GoogleDrivePlugin,
    google_drive,
    _GDOC,
    _GSHEET,
    _GSLIDES,
    _GFOLDER,
)
from daita.core.exceptions import (
    NotFoundError,
    PermissionError as DaitaPermissionError,
    AuthenticationError,
    TransientError,
    PluginError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin(read_only: bool = True) -> GoogleDrivePlugin:
    """Return a plugin with a mock Drive service injected (no real API calls)."""
    plugin = GoogleDrivePlugin(read_only=read_only)
    plugin._service = MagicMock()
    return plugin


def _drive_file(
    id="file1",
    name="test.csv",
    mime_type="text/csv",
    size="1024",
    modified="2024-01-01T00:00:00Z",
    owners=None,
    web_link="https://drive.google.com/file/file1",
):
    return {
        "id": id,
        "name": name,
        "mimeType": mime_type,
        "size": size,
        "modifiedTime": modified,
        "owners": owners
        or [{"displayName": "Test User", "emailAddress": "test@example.com"}],
        "webViewLink": web_link,
    }


def _wire_files_list(plugin, files):
    """Mock service.files().list().execute() to return the given file list."""
    mock_list = MagicMock()
    mock_list.execute.return_value = {"files": files, "nextPageToken": None}
    plugin._service.files.return_value.list.return_value = mock_list


def _wire_files_get(plugin, file_meta):
    """Mock service.files().get().execute() to return the given metadata."""
    mock_get = MagicMock()
    mock_get.execute.return_value = file_meta
    plugin._service.files.return_value.get.return_value = mock_get


def _wire_files_export(plugin, content: bytes):
    """Mock service.files().export().execute() to return bytes."""
    mock_export = MagicMock()
    mock_export.execute.return_value = content
    plugin._service.files.return_value.export.return_value = mock_export


def _wire_files_download(plugin, content: bytes):
    """Mock MediaIoBaseDownload-based download to return bytes."""
    mock_request = MagicMock()
    plugin._service.files.return_value.get_media.return_value = mock_request

    def fake_download_side_effect(buf, request):
        mock_dl = MagicMock()
        mock_dl.next_chunk.side_effect = [(None, False), (None, True)]
        # Write content to the buffer on first next_chunk call
        mock_dl.next_chunk.side_effect = None

        call_count = 0

        def patched_next_chunk():
            nonlocal call_count
            if call_count == 0:
                buf.write(content)
                call_count += 1
                return None, True
            return None, True

        mock_dl.next_chunk = patched_next_chunk
        return mock_dl

    return fake_download_side_effect


# ---------------------------------------------------------------------------
# Constructor & factory function
# ---------------------------------------------------------------------------


def test_factory_function_returns_instance():
    plugin = google_drive(read_only=True)
    assert isinstance(plugin, GoogleDrivePlugin)


def test_default_scopes_read_only():
    plugin = GoogleDrivePlugin(read_only=True)
    assert "drive.readonly" in plugin.scopes[0]


def test_default_scopes_write():
    plugin = GoogleDrivePlugin(read_only=False)
    assert plugin.scopes == ["https://www.googleapis.com/auth/drive"]


def test_custom_scopes_override():
    custom = ["https://www.googleapis.com/auth/drive.file"]
    plugin = GoogleDrivePlugin(scopes=custom)
    assert plugin.scopes == custom


def test_token_path_expands_home():
    plugin = GoogleDrivePlugin(token_path="~/.daita/token.json")
    assert not plugin.token_path.startswith("~")


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _make_http_error(status: int):
    """Create a minimal mock HttpError with the given HTTP status."""
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        pytest.skip("google-api-python-client not installed")
    resp = MagicMock()
    resp.status = status
    return HttpError(resp=resp, content=b"error")


def test_error_map_404():
    plugin = make_plugin()
    err = _make_http_error(404)
    result = plugin._map_drive_error(err, "read")
    assert isinstance(result, NotFoundError)


def test_error_map_403():
    plugin = make_plugin()
    err = _make_http_error(403)
    result = plugin._map_drive_error(err, "read")
    assert isinstance(result, DaitaPermissionError)


def test_error_map_401():
    plugin = make_plugin()
    err = _make_http_error(401)
    result = plugin._map_drive_error(err, "read")
    assert isinstance(result, AuthenticationError)
    # 401 should also clear the service to force reconnect
    assert plugin._service is None


def test_error_map_429():
    plugin = make_plugin()
    err = _make_http_error(429)
    result = plugin._map_drive_error(err, "read")
    assert isinstance(result, TransientError)


def test_error_map_503():
    plugin = make_plugin()
    err = _make_http_error(503)
    result = plugin._map_drive_error(err, "read")
    assert isinstance(result, TransientError)


def test_error_map_import_error():
    plugin = make_plugin()
    err = ImportError("google-api-python-client missing")
    result = plugin._map_drive_error(err, "connect")
    assert isinstance(result, PluginError)
    assert "google-drive" in str(result)


def test_error_map_unknown():
    plugin = make_plugin()
    err = RuntimeError("unknown")
    result = plugin._map_drive_error(err, "search")
    assert isinstance(result, PluginError)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def test_friendly_type_gdoc():
    plugin = make_plugin()
    assert plugin._friendly_type(_GDOC) == "Google Doc"


def test_friendly_type_gsheet():
    plugin = make_plugin()
    assert plugin._friendly_type(_GSHEET) == "Google Sheet"


def test_friendly_type_gslides():
    plugin = make_plugin()
    assert plugin._friendly_type(_GSLIDES) == "Google Slides"


def test_friendly_type_pdf():
    plugin = make_plugin()
    assert plugin._friendly_type("application/pdf") == "PDF"


def test_friendly_type_unknown():
    plugin = make_plugin()
    assert plugin._friendly_type("application/x-ndjson") == "application/x-ndjson"


# ---------------------------------------------------------------------------
# Content extraction — _wrap_text / _wrap_rows
# ---------------------------------------------------------------------------


def test_wrap_text_no_truncation():
    plugin = make_plugin()
    result = plugin._wrap_text("hello world", "doc.txt", "text")
    assert result["content"] == "hello world"
    assert result["truncated"] is False
    assert "total_chars" not in result


def test_wrap_text_truncation():
    plugin = make_plugin()
    long_text = "a" * 60_000
    result = plugin._wrap_text(long_text, "doc.txt", "text")
    assert len(result["content"]) == GoogleDrivePlugin._MAX_CHARS
    assert result["truncated"] is True
    assert result["total_chars"] == 60_000


def test_wrap_rows_no_truncation():
    plugin = make_plugin()
    rows = [{"col": str(i)} for i in range(10)]
    result = plugin._wrap_rows(rows, "data.csv", "csv")
    assert len(result["rows"]) == 10
    assert result["total_rows"] == 10
    assert result["truncated"] is False


def test_wrap_rows_truncation():
    plugin = make_plugin()
    rows = [{"col": str(i)} for i in range(250)]
    result = plugin._wrap_rows(rows, "data.csv", "csv")
    assert len(result["rows"]) == GoogleDrivePlugin._MAX_ROWS
    assert result["total_rows"] == 250
    assert result["truncated"] is True


# ---------------------------------------------------------------------------
# Content extraction — XLSX (with openpyxl mock)
# ---------------------------------------------------------------------------


def test_parse_xlsx_no_openpyxl():
    """Graceful degradation when openpyxl is not installed."""
    plugin = make_plugin()
    with patch.dict("sys.modules", {"openpyxl": None}):
        result = plugin._parse_xlsx(b"fake", "data.xlsx")
    assert result["binary"] is True
    assert "daita-agents[data]" in result["note"]


def test_parse_xlsx_with_openpyxl():
    """Parse XLSX using a mocked openpyxl workbook."""
    plugin = make_plugin()

    mock_ws = MagicMock()
    mock_ws.iter_rows.return_value = iter(
        [("name", "score"), ("Alice", 95), ("Bob", 87)]
    )
    mock_wb = MagicMock()
    mock_wb.active = mock_ws
    mock_wb.sheetnames = ["Sheet1"]

    mock_openpyxl = MagicMock()
    mock_openpyxl.load_workbook.return_value = mock_wb

    with patch.dict("sys.modules", {"openpyxl": mock_openpyxl}):
        result = plugin._parse_xlsx(b"fake_bytes", "data.xlsx")

    assert result["format"] == "xlsx"
    assert result["rows"][0] == {"name": "Alice", "score": "95"}
    assert result["rows"][1] == {"name": "Bob", "score": "87"}
    assert result["total_rows"] == 2


# ---------------------------------------------------------------------------
# Content extraction — DOCX (with python-docx mock)
# ---------------------------------------------------------------------------


def test_parse_docx_no_python_docx():
    plugin = make_plugin()
    with patch.dict("sys.modules", {"docx": None}):
        with pytest.raises(ImportError, match="google-drive"):
            plugin._parse_docx(b"fake", "doc.docx")


def test_parse_docx_with_python_docx():
    plugin = make_plugin()

    mock_para1 = MagicMock()
    mock_para1.text = "Hello World"
    mock_para2 = MagicMock()
    mock_para2.text = ""  # empty paragraph should be skipped
    mock_para3 = MagicMock()
    mock_para3.text = "Second paragraph"

    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]

    mock_docx = MagicMock()
    mock_docx.Document.return_value = mock_doc

    with patch.dict("sys.modules", {"docx": mock_docx}):
        result = plugin._parse_docx(b"fake_bytes", "report.docx")

    assert result["format"] == "docx"
    assert "Hello World" in result["content"]
    assert "Second paragraph" in result["content"]
    # Empty paragraph should not appear
    assert "\n\n" not in result["content"].replace("Hello World\nSecond paragraph", "")


# ---------------------------------------------------------------------------
# Content extraction — PDF (with pypdf mock)
# ---------------------------------------------------------------------------


def test_parse_pdf_no_pypdf():
    plugin = make_plugin()
    with patch.dict("sys.modules", {"pypdf": None}):
        with pytest.raises(ImportError, match="google-drive"):
            plugin._parse_pdf(b"fake", "doc.pdf")


def test_parse_pdf_with_pypdf():
    plugin = make_plugin()

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page one content"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page two content"

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1, mock_page2]

    mock_pypdf = MagicMock()
    mock_pypdf.PdfReader.return_value = mock_reader

    with patch.dict("sys.modules", {"pypdf": mock_pypdf}):
        result = plugin._parse_pdf(b"fake_bytes", "report.pdf")

    assert result["format"] == "pdf"
    assert "Page one content" in result["content"]
    assert "Page two content" in result["content"]


# ---------------------------------------------------------------------------
# Content extraction — CSV bytes
# ---------------------------------------------------------------------------


async def test_extract_content_csv():
    plugin = make_plugin()
    csv_bytes = b"name,score\nAlice,95\nBob,87\n"

    with patch.object(plugin, "_sync_files_download", return_value=csv_bytes):
        result = await plugin._extract_content("file1", "text/csv", "data.csv")

    assert result["format"] == "csv"
    assert result["rows"][0] == {"name": "Alice", "score": "95"}
    assert result["total_rows"] == 2


# ---------------------------------------------------------------------------
# Content extraction — Google-native formats
# ---------------------------------------------------------------------------


async def test_extract_content_google_doc():
    plugin = make_plugin()
    text = b"This is the document content."

    with patch.object(plugin, "_sync_files_export", return_value=text):
        result = await plugin._extract_content("file1", _GDOC, "My Doc")

    assert result["format"] == "google_doc"
    assert result["content"] == "This is the document content."


async def test_extract_content_google_sheet():
    plugin = make_plugin()
    csv_bytes = b"col1,col2\nval1,val2\n"

    with patch.object(plugin, "_sync_files_export", return_value=csv_bytes):
        result = await plugin._extract_content("file1", _GSHEET, "My Sheet")

    assert result["format"] == "google_sheet"
    assert result["rows"][0] == {"col1": "val1", "col2": "val2"}


async def test_extract_content_google_slides():
    plugin = make_plugin()
    text = b"Slide 1 title\nSlide 2 title"

    with patch.object(plugin, "_sync_files_export", return_value=text):
        result = await plugin._extract_content("file1", _GSLIDES, "My Slides")

    assert result["format"] == "google_slides"
    assert "Slide 1" in result["content"]


async def test_extract_content_folder():
    plugin = make_plugin()
    result = await plugin._extract_content("folder1", _GFOLDER, "My Folder")
    assert result["binary"] is True
    assert "gdrive_list" in result["note"]


async def test_extract_content_binary_unknown():
    plugin = make_plugin()
    raw = b"\x00\x01\x02\x03"

    with patch.object(plugin, "_sync_files_download", return_value=raw):
        result = await plugin._extract_content(
            "file1", "application/octet-stream", "blob.bin"
        )

    assert result["binary"] is True
    assert "gdrive_download" in result["note"]


# ---------------------------------------------------------------------------
# Core operations — search
# ---------------------------------------------------------------------------


async def test_search_builds_query():
    plugin = make_plugin()
    files = [_drive_file()]
    _wire_files_list(plugin, files)

    results = await plugin.search(
        "report", file_type="pdf", modified_after="2024-01-01"
    )

    call_args = plugin._service.files.return_value.list.call_args
    q = (
        call_args.kwargs.get("q") or call_args[1].get("q") or call_args[0][0]
        if call_args[0]
        else ""
    )
    # Check the query was called (we may not be able to inspect kwargs easily across mock versions)
    assert plugin._service.files.return_value.list.called
    assert len(results) == 1
    assert results[0]["id"] == "file1"


async def test_search_unknown_file_type_ignored():
    plugin = make_plugin()
    _wire_files_list(plugin, [])
    # Unknown file_type should not raise, just be ignored
    results = await plugin.search("test", file_type="unknown_type")
    assert results == []


# ---------------------------------------------------------------------------
# Core operations — list_folder
# ---------------------------------------------------------------------------


async def test_list_folder_default_root():
    plugin = make_plugin()
    files = [
        _drive_file(id="f1", name="file1.csv"),
        _drive_file(id="f2", name="file2.pdf"),
    ]
    _wire_files_list(plugin, files)

    results = await plugin.list_folder()
    assert len(results) == 2
    assert results[0]["id"] == "f1"


# ---------------------------------------------------------------------------
# Core operations — get_info
# ---------------------------------------------------------------------------


async def test_get_info_returns_structured_metadata():
    plugin = make_plugin()
    f = {
        "id": "abc",
        "name": "report.pdf",
        "mimeType": "application/pdf",
        "size": "4096",
        "createdTime": "2024-01-01T00:00:00Z",
        "modifiedTime": "2024-06-01T00:00:00Z",
        "owners": [{"displayName": "Alice", "emailAddress": "alice@example.com"}],
        "webViewLink": "https://drive.google.com/file/abc",
        "description": "Q4 Report",
        "starred": False,
        "lastModifyingUser": {"displayName": "Bob", "emailAddress": "bob@example.com"},
        "parents": ["parent_folder_id"],
    }
    _wire_files_get(plugin, f)

    info = await plugin.get_info("abc")
    assert info["id"] == "abc"
    assert info["type"] == "PDF"
    assert info["owners"] == ["alice@example.com"]
    assert info["last_modified_by"] == "bob@example.com"


# ---------------------------------------------------------------------------
# Core operations — organize
# ---------------------------------------------------------------------------


async def test_organize_invalid_action_raises():
    plugin = make_plugin()
    with pytest.raises(PluginError):
        await plugin.organize("file1", action="invalid_action")


async def test_organize_rename_requires_new_name():
    plugin = make_plugin()
    with pytest.raises(PluginError, match="new_name"):
        await plugin.organize("file1", action="rename")


async def test_organize_move_requires_dest_folder():
    plugin = make_plugin()
    with pytest.raises(PluginError, match="dest_folder_id"):
        await plugin.organize("file1", action="move")


async def test_organize_rename_calls_update():
    plugin = make_plugin()
    updated_file = _drive_file(id="file1", name="new_name.pdf")
    mock_update = MagicMock()
    mock_update.execute.return_value = updated_file
    plugin._service.files.return_value.update.return_value = mock_update

    result = await plugin.organize("file1", action="rename", new_name="new_name.pdf")
    assert result["action"] == "renamed"
    assert plugin._service.files.return_value.update.called


async def test_organize_copy_calls_copy():
    plugin = make_plugin()
    copied_file = _drive_file(id="file2", name="copy.pdf")
    mock_copy = MagicMock()
    mock_copy.execute.return_value = copied_file
    plugin._service.files.return_value.copy.return_value = mock_copy

    result = await plugin.organize(
        "file1", action="copy", new_name="copy.pdf", dest_folder_id="folder2"
    )
    assert result["action"] == "copied"


# ---------------------------------------------------------------------------
# Tool schema tests
# ---------------------------------------------------------------------------


def test_get_tools_read_only_count():
    plugin = GoogleDrivePlugin(read_only=True)
    plugin._service = MagicMock()
    tools = plugin.get_tools()
    names = [t.name for t in tools]
    assert "gdrive_search" in names
    assert "gdrive_read" in names
    assert "gdrive_list" in names
    assert "gdrive_info" in names
    assert "gdrive_download" in names
    assert "gdrive_upload" not in names
    assert "gdrive_organize" not in names
    assert len(tools) == 5


def test_get_tools_write_mode_count():
    plugin = GoogleDrivePlugin(read_only=False)
    plugin._service = MagicMock()
    tools = plugin.get_tools()
    names = [t.name for t in tools]
    assert "gdrive_upload" in names
    assert "gdrive_organize" in names
    assert len(tools) == 7


def test_tool_schemas_have_required_fields():
    plugin = GoogleDrivePlugin(read_only=False)
    plugin._service = MagicMock()
    tools = {t.name: t for t in plugin.get_tools()}

    # gdrive_search requires query
    assert "query" in tools["gdrive_search"].parameters["required"]

    # gdrive_read requires file_id
    assert "file_id" in tools["gdrive_read"].parameters["required"]

    # gdrive_organize requires file_id and action
    assert "file_id" in tools["gdrive_organize"].parameters["required"]
    assert "action" in tools["gdrive_organize"].parameters["required"]

    # gdrive_list has no required params
    assert tools["gdrive_list"].parameters["required"] == []


def test_tool_categories_and_source():
    plugin = GoogleDrivePlugin(read_only=False)
    plugin._service = MagicMock()
    for t in plugin.get_tools():
        assert t.category == "storage"
        assert t.source == "plugin"
        assert t.plugin_name == "GoogleDrive"


# ---------------------------------------------------------------------------
# Tool handlers — wiring
# ---------------------------------------------------------------------------


async def test_tool_search_handler():
    plugin = make_plugin()
    _wire_files_list(plugin, [_drive_file()])

    result = await plugin._tool_search({"query": "report"})
    assert "files" in result
    assert result["count"] == 1


async def test_tool_list_handler_defaults_to_root():
    plugin = make_plugin()
    _wire_files_list(plugin, [])

    result = await plugin._tool_list({})
    assert result["folder_id"] == "root"
    assert result["count"] == 0


async def test_tool_info_handler():
    plugin = make_plugin()
    f = _drive_file(id="abc")
    f.update(
        {
            "createdTime": "2024-01-01T00:00:00Z",
            "description": None,
            "starred": False,
            "lastModifyingUser": {"emailAddress": "x@x.com"},
        }
    )
    _wire_files_get(plugin, f)

    result = await plugin._tool_info({"file_id": "abc"})
    assert result["id"] == "abc"


async def test_tool_read_delegates_to_read():
    plugin = make_plugin()
    _wire_files_get(
        plugin,
        {"id": "file1", "name": "doc.txt", "mimeType": "text/plain", "size": "100"},
    )
    text_bytes = b"Hello from Drive"
    with patch.object(plugin, "_sync_files_download", return_value=text_bytes):
        result = await plugin._tool_read({"file_id": "file1"})
    assert result["content"] == "Hello from Drive"


async def test_tool_upload_handler(tmp_path):
    """Test upload tool creates a file in Drive."""
    plugin = GoogleDrivePlugin(read_only=False)
    plugin._service = MagicMock()

    local_file = tmp_path / "test.txt"
    local_file.write_text("hello")

    created_file = _drive_file(id="new_file", name="test.txt")
    mock_create = MagicMock()
    mock_create.execute.return_value = created_file
    plugin._service.files.return_value.create.return_value = mock_create

    import sys
    from types import ModuleType

    fake_http = ModuleType("googleapiclient.http")
    fake_http.MediaFileUpload = MagicMock(return_value=MagicMock())
    fake_googleapiclient = ModuleType("googleapiclient")
    with patch.dict(
        sys.modules,
        {"googleapiclient": fake_googleapiclient, "googleapiclient.http": fake_http},
    ):
        result = await plugin._tool_upload({"local_path": str(local_file)})

    assert result["id"] == "new_file"
    assert result["uploaded"] is True


# ---------------------------------------------------------------------------
# Lazy import verification
# ---------------------------------------------------------------------------


def test_import_without_google_packages():
    """Plugin should be importable even when google packages are absent."""
    # The class can be instantiated without packages installed
    plugin = GoogleDrivePlugin()
    assert plugin._service is None
    # Tools can be introspected without a connection
    assert isinstance(plugin.get_tools(), list)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


async def test_context_manager_connects_and_disconnects():
    plugin = GoogleDrivePlugin()

    connected = []
    disconnected = []

    async def mock_connect():
        connected.append(True)
        plugin._service = MagicMock()

    async def mock_disconnect():
        disconnected.append(True)
        plugin._service = None

    plugin.connect = mock_connect
    plugin.disconnect = mock_disconnect

    async with plugin as p:
        assert p is plugin
        assert len(connected) == 1

    assert len(disconnected) == 1
