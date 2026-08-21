"""Human-readable MCP management and guided read-tool admission."""

from __future__ import annotations

import re
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, Static

from daita import (
    MCPBindingState,
    MCPBindingStatus,
    MCPServerInspection,
    MCPToolSelection,
)

from ..models import PickerOption
from ..sanitization import safe_display, sanitize_terminal_text
from .confirm import ConfirmScreen
from .selection import SelectionScreen


@dataclass(frozen=True, slots=True)
class MCPServerGroup:
    """One presentation-only server group over independently keyed bindings."""

    local_label: str
    server_version: str
    endpoint: str
    status_label: str
    tool_names: tuple[str, ...]
    statuses: tuple[MCPBindingStatus, ...]
    stale_reasons: tuple[str, ...]


def mcp_binding_status_label(status: MCPBindingStatus) -> str:
    """Return one non-overlapping operator-facing binding state."""

    if status.binding.state is MCPBindingState.STALE:
        return "Needs refresh"
    if status.binding.state is MCPBindingState.REVOKED:
        return "Revoked"
    if status.reopen_required:
        return "Restart required"
    if status.active_in_runtime:
        return "Accepted (validated at call)"
    return "Unavailable"


def group_mcp_servers(
    statuses: tuple[MCPBindingStatus, ...],
) -> tuple[MCPServerGroup, ...]:
    """Group legacy and current bindings for display without changing identity."""

    grouped: dict[tuple[str, str], list[MCPBindingStatus]] = {}
    for status in statuses:
        key = (status.binding.local_label, status.binding.endpoint)
        grouped.setdefault(key, []).append(status)

    presentations: list[MCPServerGroup] = []
    for (local_label, endpoint), members in grouped.items():
        ordered = tuple(sorted(members, key=lambda item: item.binding.binding_id))
        if any(member.binding.state is MCPBindingState.STALE for member in ordered):
            label = "Needs refresh"
        elif any(member.reopen_required for member in ordered):
            label = "Restart required"
        elif any(member.active_in_runtime for member in ordered):
            label = "Accepted (validated at call)"
        elif all(member.binding.state is MCPBindingState.REVOKED for member in ordered):
            label = "Revoked"
        else:
            label = "Unavailable"
        presentations.append(
            MCPServerGroup(
                local_label=local_label,
                server_version=ordered[0].binding.server_version,
                endpoint=endpoint,
                status_label=label,
                tool_names=tuple(
                    sorted(
                        {
                            tool.remote_name
                            for member in ordered
                            for tool in member.binding.tools
                        },
                        key=str.casefold,
                    )
                ),
                statuses=ordered,
                stale_reasons=tuple(
                    sorted(
                        {
                            reason
                            for member in ordered
                            if (reason := member.binding.stale_reason) is not None
                        }
                    )
                ),
            )
        )
    return tuple(
        sorted(
            presentations,
            key=lambda item: (
                item.local_label.casefold(),
                item.endpoint.casefold(),
            ),
        )
    )


def render_mcp_servers(statuses: tuple[MCPBindingStatus, ...]) -> tuple[str, str]:
    """Render a compact summary and server-oriented body."""

    groups = group_mcp_servers(statuses)
    if not groups:
        return (
            "No MCP servers",
            "No remote MCP read tools are connected.\n\n"
            "Choose Add server to inspect an endpoint and select tools.",
        )
    tool_count = sum(len(group.tool_names) for group in groups)
    server_noun = "server" if len(groups) == 1 else "servers"
    tool_noun = "tool" if tool_count == 1 else "tools"
    summary = f"{len(groups)} {server_noun}  ·  {tool_count} {tool_noun}"
    blocks: list[str] = []
    for group in groups:
        name = safe_display(group.local_label, fallback="MCP server", maximum=256)
        version = safe_display(group.server_version, fallback="", maximum=256)
        heading = name + (f" {version}" if version else "")
        lines = [
            f"{heading}  ·  {group.status_label}",
            safe_display(
                group.endpoint, fallback="Endpoint unavailable", maximum=2_048
            ),
            f"{len(group.tool_names)} "
            + ("tool" if len(group.tool_names) == 1 else "tools"),
        ]
        lines.extend(
            "  • " + safe_display(tool, fallback="tool", maximum=256)
            for tool in group.tool_names
        )
        lines.extend(
            "  "
            + safe_display(reason, fallback="Remote definition changed", maximum=512)
            for reason in group.stale_reasons
        )
        blocks.append("\n".join(lines))
    return summary, "\n\n".join(blocks)


def generated_mcp_aliases(remote_names: tuple[str, ...]) -> tuple[str, ...]:
    """Create deterministic provider-safe aliases without server-specific rules."""

    aliases: list[str] = []
    used: set[str] = set()
    for remote_name in remote_names:
        base = re.sub(r"[^a-z0-9]+", "_", remote_name.casefold()).strip("_")
        if not base:
            base = "remote_tool"
        if not base[0].isalpha() or not base[0].isascii():
            base = "tool_" + base
        base = base[:40].rstrip("_") or "remote_tool"
        candidate = base
        suffix_number = 2
        while candidate in used:
            suffix = f"_{suffix_number}"
            stem = base[: 40 - len(suffix)].rstrip("_") or "tool"
            candidate = stem + suffix
            suffix_number += 1
        used.add(candidate)
        aliases.append(candidate)
    return tuple(aliases)


def mcp_tool_selections(remote_names: tuple[str, ...]) -> tuple[MCPToolSelection, ...]:
    """Build the code-owned admission records shown in the review step."""

    aliases = generated_mcp_aliases(remote_names)
    return tuple(
        MCPToolSelection(
            remote_name=remote_name,
            local_alias=alias,
            description=f"Read the explicitly admitted MCP tool {remote_name}.",
        )
        for remote_name, alias in zip(remote_names, aliases, strict=True)
    )


class MCPManagementScreen(ModalScreen[str | None]):
    """Manage remote MCP servers without exposing binding IDs as the primary UX."""

    BINDINGS = [Binding("escape", "close", "Back", priority=True)]

    def __init__(self) -> None:
        super().__init__()
        self._statuses: tuple[MCPBindingStatus, ...] = ()
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-management"):
            yield Label("MCP servers", id="mcp-title", markup=False)
            yield Static("Loading…", id="mcp-summary", markup=False)
            with VerticalScroll(id="mcp-list"):
                yield Static("", id="mcp-body", markup=False)
            yield Static(
                "Only explicitly selected, operator-attested read tools are admitted.",
                id="mcp-help",
                markup=False,
            )
            with Horizontal(id="mcp-actions"):
                yield Button("Add server", id="mcp-add", variant="primary")
                yield Button("Refresh", id="mcp-refresh")
                yield Button("Revoke", id="mcp-revoke")
                yield Button("Restart now", id="mcp-restart", disabled=True)
                yield Button("Close", id="mcp-close")
            yield Static("", id="mcp-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self.run_worker(
            self._load(),
            name="mcp-load",
            group="mcp-interaction",
            exclusive=True,
        )

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(
                "restart_required"
                if any(status.reopen_required for status in self._statuses)
                else None
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "mcp-close":
            self.action_close()
            return
        if button_id is None or self._busy:
            return
        self.run_worker(
            self._handle_button(button_id),
            name="mcp-action",
            group="mcp-interaction",
            exclusive=True,
        )

    async def _load(self) -> None:
        try:
            self._statuses = await self.app.controller.list_mcp_servers()  # type: ignore[attr-defined]
            self._render_statuses()
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)

    async def _handle_button(self, button_id: str) -> None:
        self._set_busy(True)
        self.query_one("#mcp-error", Static).update("")
        try:
            if button_id == "mcp-add":
                result = await self.app._await_modal(MCPSetupScreen())  # type: ignore[attr-defined]
                if result == "reopen":
                    self.dismiss("reopen")
                    return
                if result == "restart_required":
                    await self._load()
                    self.query_one("#mcp-help", Static).update(
                        "Server attached. Restart the agent runtime before using its tools."
                    )
                return
            if button_id == "mcp-refresh":
                await self._refresh_binding()
                return
            if button_id == "mcp-revoke":
                await self._revoke_binding()
                return
            if button_id == "mcp-restart":
                if any(status.reopen_required for status in self._statuses):
                    self.dismiss("reopen")
                else:
                    self.query_one("#mcp-help", Static).update(
                        "All current MCP tools are already active."
                    )
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._set_busy(False)

    async def _refresh_binding(self) -> None:
        status = await self._pick_binding("Refresh MCP tools", include_revoked=False)
        if status is None:
            return
        refreshed = await self.app.controller.refresh_mcp_server(  # type: ignore[attr-defined]
            status.binding.binding_id
        )
        await self._load()
        if refreshed.reopen_required:
            restart = await self.app._await_modal(  # type: ignore[attr-defined]
                ConfirmScreen(
                    "The MCP definition is current. Restart the agent runtime now "
                    "to activate this refreshed tool set?"
                )
            )
            if restart:
                self.dismiss("reopen")
                return
            self.query_one("#mcp-help", Static).update(
                "Tools refreshed. Restart the agent runtime before using them."
            )
            return
        reason = refreshed.binding.stale_reason or "The remote definition changed."
        self.query_one("#mcp-help", Static).update(
            "Tools were not activated: "
            + safe_display(reason, fallback="remote definition changed", maximum=512)
        )

    async def _revoke_binding(self) -> None:
        status = await self._pick_binding("Revoke MCP tools", include_revoked=False)
        if status is None:
            return
        binding = status.binding
        tools = ", ".join(
            safe_display(tool.remote_name, fallback="tool", maximum=256)
            for tool in binding.tools
        )
        accepted = await self.app._await_modal(  # type: ignore[attr-defined]
            ConfirmScreen(
                "Revoke MCP access for "
                + safe_display(binding.local_label, fallback="this server", maximum=256)
                + "?\n"
                + safe_display(
                    binding.endpoint, fallback="endpoint unavailable", maximum=2_048
                )
                + "\nTools: "
                + tools
                + "\n\nRevocation takes effect immediately."
            )
        )
        if not accepted:
            return
        await self.app.controller.revoke_mcp_server(  # type: ignore[attr-defined]
            binding.binding_id
        )
        await self._load()
        self.query_one("#mcp-help", Static).update(
            "MCP tool access revoked. The change took effect immediately."
        )

    async def _pick_binding(
        self,
        title: str,
        *,
        include_revoked: bool,
    ) -> MCPBindingStatus | None:
        eligible = tuple(
            status
            for status in self._statuses
            if include_revoked or status.binding.state is not MCPBindingState.REVOKED
        )
        if not eligible:
            self.query_one("#mcp-help", Static).update(
                "There are no current MCP tool sets for this action."
            )
            return None
        options = tuple(
            PickerOption(
                identity=status.binding.binding_id,
                label=safe_display(
                    status.binding.local_label,
                    fallback="MCP server",
                    maximum=256,
                )
                + " · "
                + ", ".join(
                    safe_display(tool.remote_name, fallback="tool", maximum=128)
                    for tool in status.binding.tools
                ),
                description=mcp_binding_status_label(status),
            )
            for status in eligible
        )
        selected = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(title=title, options=options)
        )
        if selected is None:
            return None
        selected_id = selected[0]
        return next(
            status for status in eligible if status.binding.binding_id == selected_id
        )

    def _render_statuses(self) -> None:
        summary, body = render_mcp_servers(self._statuses)
        self.query_one("#mcp-summary", Static).update(summary)
        self.query_one("#mcp-body", Static).update(body)
        self._update_actions()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_actions()

    def _update_actions(self) -> None:
        for button in self.query("#mcp-actions Button").results(Button):
            button.disabled = self._busy
        self.query_one("#mcp-restart", Button).disabled = self._busy or not any(
            status.reopen_required for status in self._statuses
        )

    def _show_error(self, error: Exception) -> None:
        self.query_one("#mcp-error", Static).update(
            sanitize_terminal_text(
                str(error),
                maximum=512,
                preserve_lines=False,
                fallback="MCP action failed.",
            )
        )


class MCPSetupScreen(ModalScreen[str | None]):
    """Inspect, select, review, and attach one no-auth MCP server."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self) -> None:
        super().__init__()
        self._inspection: MCPServerInspection | None = None
        self._inspected_input: str | None = None
        self._selections: tuple[MCPToolSelection, ...] = ()
        self._tool_picker: dict[str, str] = {}
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-setup"):
            yield Label("Add MCP server", id="mcp-title", markup=False)
            yield Static(
                "Step 1 of 3  ·  Enter a no-auth Streamable HTTP endpoint",
                id="mcp-step",
                markup=False,
            )
            yield Input(
                placeholder="https://mcp.example.com/mcp",
                id="mcp-endpoint",
            )
            with VerticalScroll(id="mcp-inspection"):
                yield Static(
                    "Inspection is read-only and does not grant tool access.",
                    id="mcp-inspection-body",
                    markup=False,
                )
            with Horizontal(id="mcp-setup-actions"):
                yield Button("Inspect", id="mcp-inspect", variant="primary")
                yield Button("Select tools", id="mcp-select", disabled=True)
                yield Button("Attach tools", id="mcp-attach", disabled=True)
                yield Button("Cancel", id="mcp-setup-cancel")
            yield Static("", id="mcp-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self.query_one("#mcp-endpoint", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "mcp-endpoint" or self._inspected_input is None:
            return
        if event.value.strip() == self._inspected_input:
            return
        self._inspection = None
        self._inspected_input = None
        self._selections = ()
        self._tool_picker = {}
        self.query_one("#mcp-step", Static).update(
            "Step 1 of 3  ·  Inspect the updated endpoint"
        )
        self.query_one("#mcp-inspection-body", Static).update(
            "Inspection is read-only and does not grant tool access."
        )
        self._update_actions()

    def action_cancel(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "mcp-setup-cancel":
            self.action_cancel()
            return
        if button_id is None or self._busy:
            return
        self.run_worker(
            self._handle_button(button_id),
            name="mcp-setup-action",
            group="mcp-setup-interaction",
            exclusive=True,
        )

    async def _handle_button(self, button_id: str) -> None:
        self._set_busy(True)
        self.query_one("#mcp-error", Static).update("")
        try:
            if button_id == "mcp-inspect":
                await self._inspect()
            elif button_id == "mcp-select":
                await self._select_tools()
            elif button_id == "mcp-attach":
                await self._attach_tools()
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._set_busy(False)

    async def _inspect(self) -> None:
        endpoint = self.query_one("#mcp-endpoint", Input).value.strip()
        if not endpoint:
            raise ValueError("Enter an MCP endpoint first.")
        inspection = await self.app.controller.inspect_mcp_server(endpoint)  # type: ignore[attr-defined]
        self._inspection = inspection
        self._inspected_input = endpoint
        self._selections = ()
        supported = tuple(tool for tool in inspection.tools if tool.supported)
        unsupported = tuple(tool for tool in inspection.tools if not tool.supported)
        self._tool_picker = {
            f"mcp-tool-{index}": tool.remote_name
            for index, tool in enumerate(supported)
        }
        self.query_one("#mcp-step", Static).update(
            "Step 2 of 3  ·  Select supported read tools"
        )
        lines = [
            safe_display(inspection.server_name, fallback="Unknown server", maximum=256)
            + " "
            + safe_display(inspection.server_version, fallback="", maximum=256),
            safe_display(
                inspection.endpoint, fallback="Endpoint unavailable", maximum=2_048
            ),
            "",
            f"Supported tools: {len(supported)}",
        ]
        lines.extend(
            "  • " + safe_display(tool.remote_name, fallback="tool", maximum=256)
            for tool in supported
        )
        if unsupported:
            lines.extend(("", f"Unsupported tools: {len(unsupported)}"))
            lines.extend(
                "  • "
                + safe_display(tool.remote_name, fallback="tool", maximum=256)
                + " — "
                + safe_display(
                    tool.unsupported_reason or "unsupported schema",
                    fallback="unsupported schema",
                    maximum=512,
                )
                for tool in unsupported
            )
        if not supported:
            lines.extend(("", "This server has no tools Daita can admit."))
        self.query_one("#mcp-inspection-body", Static).update("\n".join(lines))
        self._update_actions()

    async def _select_tools(self) -> None:
        inspection = self._inspection
        if inspection is None or not self._tool_picker:
            raise ValueError("Inspect a server with supported tools first.")
        supported_by_name = {
            tool.remote_name: tool for tool in inspection.tools if tool.supported
        }
        options = tuple(
            PickerOption(
                identity=picker_id,
                label=safe_display(remote_name, fallback="tool", maximum=256),
                description="Supported schema",
            )
            for picker_id, remote_name in self._tool_picker.items()
        )
        selected = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(
                title="Select independently verified read tools",
                options=options,
                multi=True,
            )
        )
        if selected is None:
            return
        remote_names = tuple(
            self._tool_picker[picker_id]
            for picker_id in selected
            if self._tool_picker[picker_id] in supported_by_name
        )
        self._selections = mcp_tool_selections(remote_names)
        self.query_one("#mcp-step", Static).update(
            "Step 3 of 3  ·  Review aliases, descriptions, and sensitivity"
        )
        lines = [
            safe_display(
                inspection.server_name, fallback="Unknown server", maximum=256
            ),
            safe_display(
                inspection.endpoint, fallback="Endpoint unavailable", maximum=2_048
            ),
            "",
            f"Selected tools: {len(self._selections)}",
        ]
        for selection in self._selections:
            lines.extend(
                (
                    "",
                    safe_display(selection.remote_name, fallback="tool", maximum=256),
                    "  Alias: " + selection.local_alias,
                    "  Description: "
                    + safe_display(
                        selection.description, fallback="MCP read tool", maximum=512
                    ),
                    "  Result sensitivity: " + selection.result_sensitivity.value,
                )
            )
        lines.extend(
            (
                "",
                "Remote descriptions and annotations are untrusted and are not copied "
                "into these tool definitions.",
            )
        )
        self.query_one("#mcp-inspection-body", Static).update("\n".join(lines))
        self._update_actions()

    async def _attach_tools(self) -> None:
        inspection = self._inspection
        if inspection is None or not self._selections:
            raise ValueError("Select at least one supported tool first.")
        accepted = await self.app._await_modal(  # type: ignore[attr-defined]
            ConfirmScreen(
                f"Attach {len(self._selections)} selected MCP "
                + ("tool" if len(self._selections) == 1 else "tools")
                + " as read-only?\n\nOnly continue if you have independently "
                "verified every selected tool is read-only. Remote metadata does not "
                "grant authority."
            )
        )
        if not accepted:
            return
        status = await self.app.controller.attach_mcp_tools(  # type: ignore[attr-defined]
            inspection.endpoint,
            self._selections,
        )
        restart = False
        if status.reopen_required:
            restart = await self.app._await_modal(  # type: ignore[attr-defined]
                ConfirmScreen(
                    "The MCP server is attached. Restart the agent runtime now to "
                    "activate the selected tools?"
                )
            )
        self.dismiss("reopen" if restart else "restart_required")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_actions()

    def _update_actions(self) -> None:
        supported = bool(self._tool_picker)
        selected = bool(self._selections)
        self.query_one("#mcp-inspect", Button).disabled = self._busy
        self.query_one("#mcp-select", Button).disabled = self._busy or not supported
        self.query_one("#mcp-attach", Button).disabled = self._busy or not selected
        self.query_one("#mcp-setup-cancel", Button).disabled = self._busy

    def _show_error(self, error: Exception) -> None:
        self.query_one("#mcp-error", Static).update(
            sanitize_terminal_text(
                str(error),
                maximum=512,
                preserve_lines=False,
                fallback="MCP setup failed.",
            )
        )
