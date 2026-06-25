"""
Extension declarations for GoogleDrivePlugin.
"""

from __future__ import annotations

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    ToolView,
)

from .manifest import PluginKind, PluginManifest

GOOGLE_DRIVE_MANIFEST = PluginManifest(
    id="google_drive",
    display_name="Google Drive",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"google_drive", "file_storage", "storage"}),
    provides=frozenset({"file_storage", "documents"}),
    optional_dependencies=frozenset({"google-api-python-client"}),
)


GOOGLE_DRIVE_OPERATION_DEFINITIONS = (
    {
        "tool_name": "gdrive_search",
        "capability_id": "google_drive.file.search",
        "operation_type": "google_drive.file.search",
        "description": "Search Google Drive for files by name, type, or date.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text to match against file names",
                },
                "file_type": {
                    "type": "string",
                    "description": (
                        "Filter by type: document, spreadsheet, presentation, "
                        "pdf, csv, docx, xlsx, image, folder"
                    ),
                },
                "folder_id": {
                    "type": "string",
                    "description": "Restrict search to files within this folder ID",
                },
                "modified_after": {
                    "type": "string",
                    "description": "ISO date string to filter recently modified files",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max files to return",
                },
            },
            "required": ["query"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_search",
    },
    {
        "tool_name": "gdrive_read",
        "capability_id": "google_drive.file.read",
        "operation_type": "google_drive.file.read",
        "description": "Read content from a Google Drive file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Google Drive file ID",
                },
                "sheet_name": {
                    "type": "string",
                    "description": "For XLSX files, the sheet name to read",
                },
            },
            "required": ["file_id"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 120,
        "handler_name": "_tool_read",
    },
    {
        "tool_name": "gdrive_list",
        "capability_id": "google_drive.folder.list",
        "operation_type": "google_drive.folder.list",
        "description": "List files in a Google Drive folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "folder_id": {
                    "type": "string",
                    "description": "Folder ID to list",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max files to return",
                },
            },
            "required": [],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_list",
    },
    {
        "tool_name": "gdrive_info",
        "capability_id": "google_drive.file.info",
        "operation_type": "google_drive.file.info",
        "description": "Get detailed metadata about a Google Drive file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Google Drive file ID",
                },
            },
            "required": ["file_id"],
        },
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 15,
        "handler_name": "_tool_info",
    },
    {
        "tool_name": "gdrive_download",
        "capability_id": "google_drive.file.download",
        "operation_type": "google_drive.file.download",
        "description": "Download a Google Drive file to a local path.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Google Drive file ID",
                },
                "local_path": {
                    "type": "string",
                    "description": "Local file path to save to",
                },
            },
            "required": ["file_id", "local_path"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.MEDIUM,
        "read_only": True,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 120,
        "handler_name": "_tool_download",
    },
    {
        "tool_name": "gdrive_upload",
        "capability_id": "google_drive.file.upload",
        "operation_type": "google_drive.file.upload",
        "description": "Upload a local file to Google Drive.",
        "parameters": {
            "type": "object",
            "properties": {
                "local_path": {
                    "type": "string",
                    "description": "Local file path to upload",
                },
                "folder_id": {
                    "type": "string",
                    "description": "Destination folder ID",
                },
                "name": {
                    "type": "string",
                    "description": "Override file name in Drive",
                },
            },
            "required": ["local_path"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 120,
        "handler_name": "_tool_upload",
    },
    {
        "tool_name": "gdrive_organize",
        "capability_id": "google_drive.file.organize",
        "operation_type": "google_drive.file.organize",
        "description": "Move, rename, or copy a file in Google Drive.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Google Drive file ID",
                },
                "action": {
                    "type": "string",
                    "enum": ["move", "rename", "copy"],
                    "description": "Action to perform on the file",
                },
                "dest_folder_id": {
                    "type": "string",
                    "description": "Destination folder ID",
                },
                "new_name": {
                    "type": "string",
                    "description": "New file name",
                },
            },
            "required": ["file_id", "action"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_organize",
    },
)


def google_drive_operation_definitions(read_only: bool = True) -> tuple[dict, ...]:
    if not read_only:
        return GOOGLE_DRIVE_OPERATION_DEFINITIONS
    return tuple(
        definition
        for definition in GOOGLE_DRIVE_OPERATION_DEFINITIONS
        if definition["read_only"]
    )


def google_drive_capabilities(read_only: bool = True) -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="google_drive",
            description=definition["description"],
            domains=frozenset({"google_drive", "file_storage", "storage"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"google_drive.operation.result"}),
            executor="google_drive.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in google_drive_operation_definitions(read_only)
    )


def google_drive_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="google_drive.operation.result",
            owner="google_drive",
            json_schema={"type": "object"},
            description="Google Drive file-storage operation result.",
        ),
    )


def google_drive_tool_views(read_only: bool = True) -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in google_drive_operation_definitions(read_only)
    )


class GoogleDriveExecutor:
    """Execute Google Drive runtime capabilities and return typed evidence."""

    id = "google_drive.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in google_drive_operation_definitions(self._plugin.read_only)
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="google_drive.operation.result",
                owner="google_drive",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "operation": definition["operation_type"],
                    "request": dict(task.input or {}),
                    "result": result,
                },
                metadata={"capability_id": task.capability_id},
            )
        ]
