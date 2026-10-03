"""List MCP servers and guide transport, authentication, and tool admission."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, Select, SelectionList, Static
from textual.widgets.selection_list import Selection

from daita import (
    MCPAuthentication,
    MCPBindingState,
    MCPBindingStatus,
    MCPCompletionSemantics,
    MCPError,
    MCPServerInspection,
    MCPToolSelection,
)
from daita.capabilities import AccessMode, AutomationEligibility, OperationalEffect
from daita.llm.models import ModelSensitivity
from daita.security import SecretReference

from ..models import PickerOption
from ..sanitization import safe_display, sanitize_terminal_text
from .confirm import ConfirmScreen
from .selection import SelectionScreen


@dataclass(frozen=True, slots=True)
class MCPServerGroup:
    """One presentation-only server group over independently keyed bindings."""

    local_label: str
    server_version: str | None
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
        return "Activation pending"
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
            label = "Activation pending"
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
            "No remote MCP tools are connected.\n\n"
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
                "Choose which tools your agent can use and review their permissions.",
                id="mcp-help",
                markup=False,
            )
            with Horizontal(id="mcp-actions"):
                yield Button("Add server", id="mcp-add", variant="primary")
                yield Button("Refresh", id="mcp-refresh")
                yield Button("Revoke", id="mcp-revoke")
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
            self.dismiss(None)

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
                if result is not None:
                    self.dismiss(result)
                return
            if button_id == "mcp-refresh":
                await self._refresh_binding()
                return
            if button_id == "mcp-revoke":
                await self._revoke_binding()
                return
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
        if refreshed.active_in_runtime:
            self.query_one("#mcp-help", Static).update(
                "MCP tools refreshed and active."
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

    def _show_error(self, error: Exception) -> None:
        self.query_one("#mcp-error", Static).update(
            sanitize_terminal_text(
                str(error),
                maximum=512,
                preserve_lines=False,
                fallback="MCP action failed.",
            )
        )


class MCPToolAdmissionScreen(ModalScreen[MCPToolSelection | None]):
    """Explicit local authority and information-handling controls for one tool."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, selection: MCPToolSelection) -> None:
        super().__init__()
        self._selection = selection

    def compose(self) -> ComposeResult:
        selected = self._selection
        with Vertical(id="mcp-tool-admission", classes="control-panel"):
            yield Label(
                safe_display(selected.remote_name, fallback="MCP tool"), markup=False
            )
            with VerticalScroll():
                yield Label("Local alias")
                yield Input(selected.local_alias, id="mcp-tool-alias")
                yield Label("Description (untrusted tool guidance; editable)")
                yield Input(selected.description, id="mcp-tool-description")
                yield Label("Data access")
                yield Select(
                    [(item.value, item.value) for item in AccessMode],
                    value=selected.access_mode.value,
                    allow_blank=False,
                    id="mcp-tool-access",
                )
                yield Label("Operational effect (independently verify the tool)")
                yield Select(
                    [
                        (item.value, item.value)
                        for item in (
                            OperationalEffect.NONE,
                            OperationalEffect.EXTERNAL_ACTION,
                            OperationalEffect.MUTATE_DATA,
                        )
                    ],
                    value=selected.operational_effect.value,
                    allow_blank=False,
                    id="mcp-tool-effect",
                )
                yield Label("Unattended eligibility (still requires a standing grant)")
                yield Select(
                    [(item.value, item.value) for item in AutomationEligibility],
                    value=cast(
                        AutomationEligibility, selected.automation_eligibility
                    ).value,
                    allow_blank=False,
                    id="mcp-tool-eligibility",
                )
                yield Label("Result sensitivity")
                yield Select(
                    [(item.value, item.value) for item in ModelSensitivity],
                    value=selected.result_sensitivity.value,
                    allow_blank=False,
                    id="mcp-tool-result",
                )
                yield Label("Tool outbound ceiling (also bounded by server ceiling)")
                yield Select(
                    [(item.value, item.value) for item in ModelSensitivity],
                    value=selected.maximum_outbound_sensitivity.value,
                    allow_blank=False,
                    id="mcp-tool-outbound",
                )
                yield Label("Known completion semantics")
                yield Select(
                    [(item.value, item.value) for item in MCPCompletionSemantics],
                    value=selected.completion_semantics.value,
                    allow_blank=False,
                    id="mcp-tool-completion",
                )
                yield Static(
                    "Direct results prove server-reported invocation only. Fix nested arguments in full in standing grants. Shell, infrastructure, arbitrary execution and asynchronous completion are unsupported.",
                    markup=False,
                )
            yield Static("", id="mcp-admission-error", markup=False)
            yield Button(
                "Save local permissions", id="mcp-admission-save", variant="primary"
            )
            yield Button("Cancel", id="mcp-admission-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-admission-cancel":
            self.dismiss(None)
        elif event.button.id == "mcp-admission-save":
            try:
                values = {
                    name: cast(str, self.query_one(f"#mcp-tool-{name}", Select).value)
                    for name in (
                        "access",
                        "effect",
                        "eligibility",
                        "result",
                        "outbound",
                        "completion",
                    )
                }
                self.dismiss(
                    replace(
                        self._selection,
                        local_alias=self.query_one("#mcp-tool-alias", Input).value,
                        description=self.query_one(
                            "#mcp-tool-description", Input
                        ).value,
                        access_mode=AccessMode(values["access"]),
                        operational_effect=OperationalEffect(values["effect"]),
                        automation_eligibility=AutomationEligibility(
                            values["eligibility"]
                        ),
                        result_sensitivity=ModelSensitivity(values["result"]),
                        maximum_outbound_sensitivity=ModelSensitivity(
                            values["outbound"]
                        ),
                        completion_semantics=MCPCompletionSemantics(
                            values["completion"]
                        ),
                    )
                )
            except (TypeError, ValueError) as error:
                self.query_one("#mcp-admission-error", Static).update(str(error))


class MCPSetupScreen(ModalScreen[str | None]):
    """Connect, review exact tool permissions, and report activation."""

    BINDINGS = [Binding("escape", "cancel", "Back / Cancel", priority=True)]

    def __init__(self) -> None:
        super().__init__()
        self._inspection: MCPServerInspection | None = None
        self._candidates: dict[str, MCPToolSelection] = {}
        self._busy = False
        self._step = "connect"
        self._authentication = MCPAuthentication.no_auth()
        self._owned_credential: SecretReference | None = None
        self._attached = False
        self._result: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-setup"):
            yield Label("Add MCP server", id="mcp-title", markup=False)
            yield Static("", id="mcp-step", markup=False)
            with VerticalScroll(id="mcp-connect"):
                yield Label("Server URL")
                yield Input(
                    placeholder="https://mcp.example.com/mcp", id="mcp-endpoint"
                )
                yield Label("Authentication")
                yield Select(
                    [
                        ("None", "none"),
                        ("API key / bearer token", "token"),
                        ("Advanced: credential reference", "reference"),
                    ],
                    value="none",
                    allow_blank=False,
                    id="mcp-auth",
                )
                yield Input(
                    placeholder="Paste key or token without the Bearer prefix",
                    password=True,
                    id="mcp-token",
                )
                with Vertical(id="mcp-reference-fields"):
                    yield Select(
                        [("Environment variable", "env"), ("Keychain", "keychain")],
                        value="env",
                        allow_blank=False,
                        id="mcp-reference-kind",
                    )
                    yield Input(
                        placeholder="Credential reference name", id="mcp-credential-ref"
                    )
                yield Static("", id="mcp-credential-status", markup=False)
                yield Static(
                    "Keys and tokens are saved in your local keychain. Browser sign-in "
                    "is not supported here. For API keys, use the server's API-key endpoint.\n"
                    "Finding tools does not give your agent access to them.",
                    id="mcp-auth-help",
                    markup=False,
                )
            with VerticalScroll(id="mcp-review"):
                yield Static("", id="mcp-inspection-body", markup=False)
                yield SelectionList[str](id="mcp-tools")
                with Horizontal(id="mcp-tool-actions"):
                    yield Button("Advanced / permissions", id="mcp-configure")
                    yield Static("0 selected", id="mcp-selection-count", markup=False)
                yield Label("What data may be sent to this server?")
                yield Select(
                    [
                        ("Public only", "public"),
                        ("Public and internal", "internal"),
                        ("Up to confidential", "confidential"),
                        ("Up to restricted (all levels)", "restricted"),
                    ],
                    value=ModelSensitivity.INTERNAL.value,
                    allow_blank=False,
                    id="mcp-outbound",
                )
                yield Static(
                    "Verify each tool's access before adding. Defaults assume reads.\n"
                    "Data limits cover the full request; stricter tool limits apply.",
                    id="mcp-review-help",
                    markup=False,
                )
            with VerticalScroll(id="mcp-finished"):
                yield Static("", id="mcp-success", markup=False)
            yield Static("", id="mcp-error", markup=False)
            with Horizontal(id="mcp-setup-actions"):
                yield Button("Find tools", id="mcp-inspect", variant="primary")
                yield Button("Add server", id="mcp-attach", variant="primary")
                yield Button("Back", id="mcp-back")
                yield Button("Cancel", id="mcp-setup-cancel")
                yield Button("Return to chat", id="mcp-done", variant="primary")
            yield Footer()

    def on_mount(self) -> None:
        self._update_auth_fields()
        self._set_step("connect")
        self.query_one("#mcp-endpoint", Input).focus()

    def _set_step(self, step: str) -> None:
        self._step = step
        self.query_one("#mcp-connect").display = step == "connect"
        self.query_one("#mcp-review").display = step == "review"
        self.query_one("#mcp-finished").display = step == "finished"
        self.query_one("#mcp-step", Static).update(
            {
                "connect": "Step 1 of 2 · Connect to a server",
                "review": "Step 2 of 2 · Choose tools and review access",
                "finished": (
                    "Server added" if self._result == "active" else "Server saved"
                ),
            }[step]
        )
        self._update_actions()

    def _update_auth_fields(self) -> None:
        mode = self.query_one("#mcp-auth", Select).value
        self.query_one("#mcp-token", Input).display = mode == "token"
        self.query_one("#mcp-reference-fields").display = mode == "reference"
        self.query_one("#mcp-credential-status", Static).update(
            "Credential saved. Leave the token field empty to keep it."
            if mode == "token" and self._owned_credential is not None
            else ""
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in {"mcp-endpoint", "mcp-credential-ref", "mcp-token"}:
            self._clear_inspection()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in {"mcp-auth", "mcp-reference-kind"}:
            self._clear_inspection()
            self._update_auth_fields()

    def _clear_inspection(self) -> None:
        if self._inspection is None or self._attached:
            return
        self._inspection = None
        self._candidates = {}
        self.query_one("#mcp-tools", SelectionList).clear_options()
        self._set_step("connect")

    def action_cancel(self) -> None:
        if self._busy:
            return
        if self._attached:
            self.dismiss(self._result)
        elif self._step == "review":
            self._set_step("connect")
            self.query_one("#mcp-endpoint", Input).focus()
        else:
            self._set_busy(True)
            self.run_worker(self._handle_button("mcp-setup-cancel"), name="mcp-cancel")

    async def _cancel_setup(self) -> None:
        await self._discard_unused_credential()
        self.dismiss(None)

    async def _discard_unused_credential(self) -> None:
        if self._owned_credential is not None and not self._attached:
            await self.app.controller.delete_mcp_bearer(self._owned_credential)  # type: ignore[attr-defined]
            self._owned_credential = None

    async def on_unmount(self) -> None:
        await self._discard_unused_credential()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id is None or self._busy:
            return
        if button_id == "mcp-done":
            self.dismiss(self._result)
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
            elif button_id == "mcp-attach":
                await self._attach_tools()
            elif button_id == "mcp-configure":
                await self._configure_tool()
            elif button_id == "mcp-back":
                self._set_step("connect")
            elif button_id == "mcp-setup-cancel":
                await self._cancel_setup()
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._set_busy(False)
        if self.is_mounted:
            if self._step == "finished":
                self.query_one("#mcp-done", Button).focus()
            elif button_id in {"mcp-inspect", "mcp-back", "mcp-configure"}:
                self.query_one(
                    "#mcp-tools" if self._step == "review" else "#mcp-endpoint"
                ).focus()

    async def _inspect(self) -> None:
        endpoint = self.query_one("#mcp-endpoint", Input).value.strip()
        if not endpoint:
            raise ValueError("Enter an MCP server URL first.")
        self._clear_inspection()
        mode = self.query_one("#mcp-auth", Select).value
        if mode == "none":
            await self._discard_unused_credential()
            authentication = MCPAuthentication.no_auth()
        elif mode == "reference":
            await self._discard_unused_credential()
            name = self.query_one("#mcp-credential-ref", Input).value.strip()
            scheme = cast(str, self.query_one("#mcp-reference-kind", Select).value)
            authentication = MCPAuthentication.bearer(SecretReference(scheme, name))
        else:
            token_input = self.query_one("#mcp-token", Input)
            if token_input.value:
                if token_input.value.lower().startswith("bearer "):
                    raise ValueError(
                        "Enter only the key or token, without the Bearer prefix."
                    )
                await self._discard_unused_credential()
                self._owned_credential = await self.app.controller.store_mcp_bearer(token_input.value)  # type: ignore[attr-defined]
                with token_input.prevent(Input.Changed):
                    token_input.value = ""
            if self._owned_credential is None:
                raise ValueError("Enter an API key or bearer token first.")
            authentication = MCPAuthentication.bearer(self._owned_credential)
        self._update_auth_fields()
        inspection = await self.app.controller.inspect_mcp_server(endpoint, authentication=authentication)  # type: ignore[attr-defined]
        self._authentication = authentication
        self._inspection = inspection
        supported = tuple(tool for tool in inspection.tools if tool.supported)
        self._candidates = {
            item.remote_name: item
            for item in MCPToolSelection.resolve(
                tuple(MCPToolSelection(tool.remote_name) for tool in supported),
                inspection,
            )
        }
        listing = self.query_one("#mcp-tools", SelectionList)
        listing.clear_options()
        for tool in inspection.tools:
            candidate = self._candidates.get(tool.remote_name)
            if candidate is not None:
                listing.add_option(
                    Selection(self._tool_prompt(candidate), tool.remote_name)
                )
            else:
                reason = safe_display(
                    tool.unsupported_reason or "unsupported schema", maximum=512
                )
                name = safe_display(tool.remote_name, fallback="tool", maximum=256)
                listing.add_option(
                    Selection(
                        Text(f"{name} · Unavailable: {reason}"),
                        tool.remote_name,
                        disabled=True,
                    )
                )
        summary = (
            safe_display(inspection.server_name, fallback="MCP server", maximum=256)
            + " "
            + safe_display(inspection.server_version, fallback="", maximum=256)
            + f" · {len(supported)} tools available"
            + "\n"
            + safe_display(inspection.endpoint, fallback="MCP endpoint", maximum=2_048)
        )
        if not supported:
            summary += "\nNo supported tools. Go back to use another endpoint."
        self.query_one("#mcp-inspection-body", Static).update(summary)
        listing.highlighted = next(
            (index for index, tool in enumerate(inspection.tools) if tool.supported),
            None,
        )
        self._set_step("review")
        self._render_selection()

    def _tool_prompt(self, selection: MCPToolSelection) -> Text:
        name = safe_display(selection.remote_name, fallback="tool", maximum=256)
        effect = selection.operational_effect
        permission = f"{selection.access_mode.value.capitalize()} access · " + (
            "No effects"
            if effect is OperationalEffect.NONE
            else effect.value.replace("_", " ")
        )
        return Text(f"{name} · {permission}")

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        if event.selection_list.id == "mcp-tools":
            self._render_selection()

    def on_selection_list_selection_highlighted(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        if event.selection_list.id == "mcp-tools":
            self._update_actions()

    def _selected_tools(self) -> tuple[MCPToolSelection, ...]:
        selected = set(self.query_one("#mcp-tools", SelectionList).selected)
        return tuple(
            item for name, item in self._candidates.items() if name in selected
        )

    async def _configure_tool(self) -> None:
        listing = self.query_one("#mcp-tools", SelectionList)
        index = listing.highlighted
        if index is None:
            return
        name = listing.get_option_at_index(index).value
        original = self._candidates.get(name)
        if original is None:
            return
        configured = await self.app._await_modal(MCPToolAdmissionScreen(original))  # type: ignore[attr-defined]
        if configured is not None:
            self._candidates[name] = configured
            listing.replace_option_prompt_at_index(index, self._tool_prompt(configured))
            self._render_selection()

    def _render_selection(self) -> None:
        count = len(self._selected_tools())
        self.query_one("#mcp-selection-count", Static).update(f"{count} selected")
        self.query_one("#mcp-attach", Button).label = f"Add server · {count} " + (
            "tool" if count == 1 else "tools"
        )
        self._update_actions()

    async def _attach_tools(self) -> None:
        inspection = self._inspection
        selections = self._selected_tools()
        if (
            self._attached
            or self._step != "review"
            or inspection is None
            or not selections
        ):
            raise ValueError("Choose at least one supported tool first.")
        status = await self.app.controller.attach_mcp_tools(  # type: ignore[attr-defined]
            inspection.endpoint,
            selections,
            authentication=self._authentication,
            maximum_outbound_sensitivity=ModelSensitivity(
                cast(str, self.query_one("#mcp-outbound", Select).value)
            ),
        )
        # Admission has persisted even if runtime activation needs attention. Never
        # offer a duplicate attachment or delete the binding's credential afterward.
        self._attached = True
        self._result = "active" if status.active_in_runtime else "saved"
        count = len(selections)
        message = (
            f"Your agent can discover and use {count} "
            + ("tool" if count == 1 else "tools")
            + " in your next message."
            if status.active_in_runtime
            else "The server is saved, but activation is pending. Check its status in MCP servers."
        )
        self.query_one("#mcp-success", Static).update(message)
        self._set_step("finished")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in self.query("Input, Select, SelectionList"):
            widget.disabled = busy or self._attached
        self._update_actions()

    def _update_actions(self) -> None:
        for button_id, visible in {
            "mcp-inspect": self._step == "connect",
            "mcp-attach": self._step == "review",
            "mcp-back": self._step == "review",
            "mcp-setup-cancel": self._step != "finished",
            "mcp-done": self._step == "finished",
        }.items():
            button = self.query_one(f"#{button_id}", Button)
            button.display = visible
            button.disabled = self._busy
        self.query_one("#mcp-inspect", Button).label = (
            "Finding tools…" if self._busy and self._step == "connect" else "Find tools"
        )
        self.query_one("#mcp-attach", Button).disabled = (
            self._busy or not self._selected_tools()
        )
        listing = self.query_one("#mcp-tools", SelectionList)
        highlighted = listing.highlighted
        self.query_one("#mcp-configure", Button).disabled = (
            self._busy
            or highlighted is None
            or listing.get_option_at_index(highlighted).value not in self._candidates
        )

    def _show_error(self, error: Exception) -> None:
        message = sanitize_terminal_text(
            str(error), maximum=512, preserve_lines=False, fallback="MCP setup failed."
        )
        if isinstance(error, MCPError) and error.code == "mcp_authentication_failed":
            message += (
                " Check the credential and endpoint. API keys may use a different "
                "URL from browser sign-in; this screen cannot sign in through a browser."
            )
        self.query_one("#mcp-error", Static).update(message)
