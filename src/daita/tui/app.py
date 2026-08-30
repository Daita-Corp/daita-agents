"""Coordinate the Textual application lifecycle, navigation, and Agent.run path."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.theme import Theme

from daita import (
    ApprovalDecision,
    ApprovalRequest,
    JobStatus,
    LocalWorkspace,
    LoopExit,
    LoopExitKind,
)
from daita.observation import AgentEventKind
from daita.security import KeychainStore

from .clipboard import deliver_clipboard
from .commands import (
    SKILL_DESCRIPTION_PLACEHOLDER,
    SKILL_INSTRUCTIONS_PLACEHOLDER,
    matching_completions,
    parse_skill_editor_document,
    render_skill_editor_document,
    source_override_completions,
)
from .controller import PresentationController
from .models import (
    MAX_COMPOSER_CHARACTERS,
    MIN_READY_ROWS,
    MIN_USABLE_COLUMNS,
    PickerOption,
    TranscriptBlock,
    UserInputError,
    validate_candidate_review_cost_limit,
)
from .observer import ObserverEvent, RunObserver
from .projection import CAPABILITY_LABELS, project_conversation
from .sanitization import render_model_answer, sanitize_terminal_text
from .screens.catalog import CatalogScreen
from .screens.chat import ChatScreen
from .screens.confirm import ConfirmScreen
from .screens.editing import ReviewCostScreen, SkillNameScreen
from .screens.inbox import InboxScreen
from .screens.jobs import JobsScreen
from .screens.mcp import MCPManagementScreen, MCPSetupScreen
from .screens.onboarding import (
    AgentCreateScreen,
    ModelSetupScreen,
    SourceSetupScreen,
)
from .screens.permissions import PermissionsScreen
from .screens.routines import RoutinesScreen
from .screens.selection import SelectionScreen
from .screens.source_edit import SourceEditScreen
from .widgets.composer import (
    ComposerExitRequested,
    ComposerLimitReached,
    ComposerSubmitted,
)
from .widgets.welcome import WelcomeView

DAITA_THEME = Theme(
    name="daita",
    primary="#ACFD21",
    secondary="#111111",
    warning="#FBBF24",
    error="#DE3535",
    success="#ACFD21",
    accent="#ACFD21",
    foreground="#FFFFFF",
    background="#000000",
    surface="#0D0D0D",
    panel="#111111",
    boost="#191C1F",
    dark=True,
    text_alpha=1.0,
    variables={
        "block-cursor-background": "#ACFD21",
        "block-cursor-foreground": "#000000",
        "block-cursor-text-style": "bold",
        "block-cursor-blurred-background": "#ACFD21",
        "block-cursor-blurred-foreground": "#000000",
        "block-cursor-blurred-text-style": "bold",
        "input-cursor-background": "#ACFD21",
        "input-cursor-foreground": "#000000",
        "input-selection-background": "#3E5A12",
        "input-selection-foreground": "#FFFFFF",
    },
)

_CREATE_NEW_AGENT_SELECTION = "daita:create-new-agent"


def _run_failure_notice(result: LoopExit) -> str:
    if result.reason == "timeout":
        return (
            "The model provider timed out after bounded retries. Daita stopped "
            "waiting; any completed tool results remain available above."
        )
    if result.reason == "wall_time_exhausted":
        return (
            "The run reached its overall time limit and was stopped. Any completed "
            "tool results remain available above."
        )
    return f"{result.kind.value}: {result.reason}"


class DaitaApp(App[int]):
    """One Textual app for onboarding, chat, commands, and approval."""

    CSS_PATH = "daita.tcss"
    TITLE = "Daita"

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        workspace: LocalWorkspace,
        agent_name: str | None = None,
        keychain: KeychainStore | None = None,
        model_validator: Any = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
        model: Any = None,
        model_profile: Any = None,
        start_bootstrap: bool = True,
    ) -> None:
        super().__init__()
        self.register_theme(DAITA_THEME)
        self.theme = DAITA_THEME.name
        validate_candidate_review_cost_limit(reviewer_max_estimated_cost_usd)
        self.controller = PresentationController(
            root=root,
            workspace=workspace,
            keychain=keychain,
            model_validator=model_validator,
            reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
        )
        self.controller.model = model
        self.controller.model_profile = model_profile
        self._requested_agent = agent_name
        self._observer = RunObserver(self)
        self._run_task: asyncio.Task[None] | None = None
        self._pending_user_identity: str | None = None
        self._partial_identity = "assistant.partial"
        self._partial_text = ""
        self._context_input_tokens: int | None = None
        self._context_conversation_id: str | None = None
        self._shutting_down = False
        self._too_small = False
        self._exit_code = 0
        self._modal_future: asyncio.Future[Any] | None = None
        self._start_bootstrap = start_bootstrap
        self._startup_error: Exception | None = None
        self._active_job_count = 0
        self._inbox_item_count = 0
        self._autonomous_run_ids: set[str] = set()
        self._known_inbox_ids: set[str] | None = None
        self._background_refresh_lock = asyncio.Lock()
        self._completion_cache: (
            tuple[
                tuple[tuple[str, str], ...],
                tuple[tuple[str, str, str], ...],
            ]
            | None
        ) = None

    def compose(self) -> ComposeResult:
        yield WelcomeView(booting=True, id="boot")

    def on_mount(self) -> None:
        self.set_interval(
            2.0,
            self._poll_background_status,
            name="daita-background-status",
        )
        if self._start_bootstrap:
            self.run_worker(self._bootstrap(), exclusive=True, name="bootstrap")

    async def _await_modal(self, screen: Any) -> Any:
        """Await a modal from any app-loop task, including Agent.run()."""

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._modal_future = future

        def _done(result: Any) -> None:
            if not future.done():
                future.set_result(result)

        self.push_screen(screen, callback=_done)
        try:
            return await future
        finally:
            if self._modal_future is future:
                self._modal_future = None

    async def on_unmount(self) -> None:
        await self._shutdown()

    async def _shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._observer.close()
        pending = self._modal_future
        if pending is not None and not pending.done():
            pending.set_result(None)
        await self._settle_run()
        await self.controller.close()

    async def _settle_run(self) -> None:
        task = self._run_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def handle_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        if self._shutting_down:
            raise asyncio.CancelledError
        if self.size.height < 15:
            raise RuntimeError("terminal is too small to review this change")
        screen = self.chat()
        if screen is None:
            raise RuntimeError("chat view is unavailable for approval review")
        decision = await screen.request_approval(request)
        if decision is None:
            raise asyncio.CancelledError
        return decision

    async def _bootstrap(self) -> None:
        try:
            await self._bootstrap_inner()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._exit_code = 1
            self._startup_error = error
            self.exit(1)

    async def _bootstrap_inner(self) -> None:
        names = await self.controller.list_agents()
        if self._requested_agent:
            await self._open(self._requested_agent)
        elif len(names) == 1:
            await self._open(names[0])
        elif names:
            while self.controller.agent is None:
                selected = await self._await_modal(
                    SelectionScreen(
                        title="Select an agent",
                        options=tuple(PickerOption(name, name) for name in names),
                        secondary_action=PickerOption(
                            _CREATE_NEW_AGENT_SELECTION,
                            "Create new agent",
                        ),
                    )
                )
                if selected is None:
                    self.exit(0)
                    return
                if selected == (_CREATE_NEW_AGENT_SELECTION,):
                    await self._await_modal(AgentCreateScreen())
                    continue
                await self._open(selected[0])
        else:
            created = await self._await_modal(AgentCreateScreen())
            if created is None:
                self.exit(0)
                return
        await self._ensure_ready()

    async def create_named_agent(self, name: str) -> None:
        self._reset_background_status()
        await self.controller.create_agent(
            name,
            observer=self._observer,
            approval_handler=self.handle_approval,
        )

    async def _open(self, name: str) -> None:
        self._reset_background_status()
        await self.controller.open_agent(
            name,
            observer=self._observer,
            approval_handler=self.handle_approval,
        )

    async def _ensure_ready(self) -> None:
        await self._show_chat()
        await self._show_home_guidance()

    async def _show_home_guidance(self) -> None:
        chat = self.chat()
        if chat is None or self.controller.agent is None:
            return
        setup_guidance: list[str] = []
        agent = self.controller.require_agent()
        if agent.model_profile is None:
            setup_guidance.append("no model · use /model")
        sources = tuple(
            source for source in await self.controller.list_sources() if source.active
        )
        workspace = self.controller.workspace
        workspace_status = (
            f"Files: {workspace.root.name} ({workspace.sensitivity.value})"
        )
        if not sources:
            source_status = "Run source: none connected (a source is optional)"
        else:
            active_source = await self.controller.active_source()
            if len(sources) > 1 and active_source is None:
                source_status = "Run source: Files only or choose with @"
            else:
                selected = active_source or next(iter(sources))
                source_status = f"Run source: {selected.display_name}"
            if (await self.controller.catalog_summary()).is_empty:
                setup_guidance.append("catalog has 0 resources · use /source edit")
        status = "  ·  ".join((workspace_status, source_status, *setup_guidance))
        chat.show_notice(status)

    async def _pick_source(self) -> None:
        sources = await self.controller.list_sources()
        options = tuple(
            PickerOption(source.id, source.display_name, source.adapter_id)
            for source in sources
            if source.active
        )
        selected = await self._await_modal(
            SelectionScreen(title="Choose the active source", options=options)
        )
        if selected is None:
            return
        await self.controller.select_source(selected[0])

    async def _show_chat(self) -> None:
        self.invalidate_completion_cache()
        await self._load_completion_cache()
        await self.push_screen(ChatScreen())
        await self._replace_conversation_transcript()
        await self._refresh_status()
        await self.refresh_background_status(notify_new=False)

    def _reset_background_status(self) -> None:
        self._active_job_count = 0
        self._inbox_item_count = 0
        self._autonomous_run_ids.clear()
        self._known_inbox_ids = None

    async def _poll_background_status(self) -> None:
        await self.refresh_background_status(notify_new=True)

    async def refresh_background_status(self, *, notify_new: bool) -> None:
        """Refresh bounded read-only background indicators for the open TUI."""

        if (
            self._shutting_down
            or self.controller.agent is None
            or self._background_refresh_lock.locked()
        ):
            return
        async with self._background_refresh_lock:
            jobs = None
            inbox = None
            try:
                jobs = await self.controller.list_jobs()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            try:
                inbox = await self.controller.list_inbox()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            if jobs is not None:
                self._active_job_count = sum(
                    item.status
                    in {
                        JobStatus.QUEUED,
                        JobStatus.RUNNING,
                        JobStatus.CANCEL_REQUESTED,
                    }
                    for item in jobs
                )
            new_delivery_ids: set[str] = set()
            if inbox is not None:
                current_ids = {item.delivery_id for item in inbox}
                self._inbox_item_count = len(current_ids)
                if self._known_inbox_ids is None:
                    self._known_inbox_ids = set(current_ids)
                else:
                    new_delivery_ids = current_ids - self._known_inbox_ids
                    self._known_inbox_ids.update(current_ids)
            await self._refresh_status()
            if notify_new and new_delivery_ids:
                count = len(new_delivery_ids)
                noun = "report is" if count == 1 else "reports are"
                self.notify(
                    f"{count} background {noun} ready. Open /inbox to review.",
                    title="Inbox",
                    timeout=8,
                )

    async def _replace_conversation_transcript(self) -> None:
        screen = self.chat()
        if screen is None:
            return
        conversation_id = self.controller.conversation_id
        if conversation_id is None:
            screen.set_blocks(())
            return
        runs = await self.controller.conversation_runs(conversation_id)
        screen.set_blocks(project_conversation(runs))

    def _reset_context_usage(self) -> None:
        self._context_input_tokens = None
        self._context_conversation_id = self.controller.conversation_id

    def invalidate_completion_cache(self) -> None:
        self._completion_cache = None

    async def _load_completion_cache(
        self,
    ) -> tuple[
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str, str], ...],
    ]:
        cached = self._completion_cache
        if cached is not None:
            return cached
        skill_matches: tuple[tuple[str, str], ...] = ()
        source_matches: tuple[tuple[str, str, str], ...] = ()
        if self.controller.agent is not None:
            try:
                skill_matches = await self.controller.skill_completions()
            except Exception:
                skill_matches = ()
            try:
                source_matches = source_override_completions(
                    await self.controller.list_sources()
                )
            except Exception:
                source_matches = ()
        cached = (skill_matches, source_matches)
        self._completion_cache = cached
        return cached

    async def completion_matches(self, prefix: str) -> tuple[tuple[str, str, str], ...]:
        if "\n" in prefix or not prefix.startswith(("/", "@")):
            return ()
        skill_matches, source_matches = await self._load_completion_cache()
        return matching_completions(
            prefix,
            skill_completions=skill_matches,
            source_completions=source_matches,
        )

    def chat(self) -> ChatScreen | None:
        if not self.screen_stack:
            return None
        if isinstance(self.screen, ChatScreen):
            return self.screen
        return None

    async def _refresh_status(
        self, *, running: bool = False, state: str = "ready"
    ) -> None:
        screen = self.chat()
        if screen is None or self.controller.agent is None:
            return
        too_small = (
            self.size.width < MIN_USABLE_COLUMNS or self.size.height < MIN_READY_ROWS
        )
        self._too_small = too_small
        profile = self.controller.require_agent().model_profile
        screen.set_status(
            agent=self.controller.require_agent().name,
            model=self.controller.model_label(),
            source=await self.controller.source_summary(),
            state=state if running else "ready",
            context_used=self._context_input_tokens,
            context_total=(
                profile.context_window_tokens if profile is not None else None
            ),
            active_jobs=self._active_job_count,
            active_reports=len(self._autonomous_run_ids),
            inbox_items=self._inbox_item_count,
            too_small=too_small,
        )
        if too_small:
            screen.set_submitting(True)
        elif not running:
            screen.set_submitting(False)

    def on_resize(self) -> None:
        self.call_later(self._refresh_status)

    def on_composer_submitted(self, event: ComposerSubmitted) -> None:
        # A command may await a modal result. Keep that wait out of the App's
        # message handler so key and mouse events can continue reaching the
        # modal screen.
        self.run_worker(
            self.submit_composer(event.text),
            name="composer-submit",
            group="foreground-interaction",
        )

    async def on_composer_limit_reached(self, _event: ComposerLimitReached) -> None:
        screen = self.chat()
        if screen is not None:
            screen.show_notice(
                f"Input is limited to {MAX_COMPOSER_CHARACTERS} characters."
            )

    def on_composer_exit_requested(self, _event: ComposerExitRequested) -> None:
        self.run_worker(
            self._request_exit(),
            name="exit-request",
            group="foreground-interaction",
        )

    async def submit_composer(self, raw: str) -> None:
        screen = self.chat()
        if screen is None or self._too_small:
            return
        if self._run_task is not None and not self._run_task.done():
            return
        text = raw.strip()
        if not text:
            return
        screen.composer().clear()
        screen.clear_notice()
        if text == "/files" or text.startswith("/files "):
            message = text.removeprefix("/files").strip()
            if not message:
                screen.show_notice("Usage: /files <question>")
                return
            await self._start_run(
                message,
                files_only=True,
                display=text,
            )
            return
        if text.startswith("/"):
            await self._handle_command(text)
            return
        try:
            message, source_id, _learn = await self.controller.prepare_message(text)
        except UserInputError as error:
            screen.show_notice(str(error))
            return
        await self._start_run(message, source_id=source_id, display=text)

    async def _handle_command(self, command: str) -> None:
        screen = self.chat()
        if screen is None:
            return
        self.invalidate_completion_cache()
        try:
            invocation = await self.controller.skill_invocation_message(command)
        except ValueError as error:
            screen.show_notice(str(error))
            return
        if invocation is not None and (
            command.startswith("/skills use") or command.split()[0] != "/skills"
        ):
            await self._start_run(invocation, display=command)
            return
        prior_conversation_id = self.controller.conversation_id
        try:
            outcome = await self.controller.dispatch_command(command)
        except (UserInputError, ValueError, RuntimeError, OSError) as error:
            screen.show_notice(str(error))
            return
        if self.controller.conversation_id != prior_conversation_id:
            self._reset_context_usage()
            await self._replace_conversation_transcript()
            await self._refresh_status()
        if outcome.kind == "exit":
            await self._request_exit()
            return
        if outcome.kind == "run" and outcome.run_message is not None:
            await self._start_run(outcome.run_message, display=command)
            return
        if outcome.kind == "notice":
            screen.append_block(
                TranscriptBlock(
                    "notice", f"notice-{len(command)}-{id(outcome)}", outcome.message
                )
            )
            return
        if outcome.kind == "screen":
            try:
                await self._open_command_screen(
                    outcome.screen,
                    outcome.payload,
                    message=outcome.message,
                )
            except (UserInputError, ValueError, RuntimeError, OSError) as error:
                screen.show_notice(
                    sanitize_terminal_text(
                        str(error),
                        maximum=512,
                        preserve_lines=False,
                        fallback="Command failed.",
                    )
                )
            return
        if outcome.kind == "confirm":
            try:
                await self._open_command_screen(
                    outcome.screen,
                    outcome.payload,
                    message=outcome.message,
                )
            except (UserInputError, ValueError, RuntimeError, OSError) as error:
                screen.show_notice(
                    sanitize_terminal_text(
                        str(error),
                        maximum=512,
                        preserve_lines=False,
                        fallback="Command failed.",
                    )
                )

    async def _open_command_screen(
        self,
        screen_name: str | None,
        payload: dict[str, Any],
        message: str = "",
    ) -> None:
        if screen_name == "source_picker":
            await self._pick_source()
            await self._refresh_status()
            await self._show_home_guidance()
            return
        if screen_name == "source_setup":
            await self._await_modal(SourceSetupScreen())
            await self._refresh_status()
            await self._show_home_guidance()
            return
        if screen_name == "source_edit":
            changed = await self._await_modal(SourceEditScreen())
            if changed:
                self.controller.conversation_id = None
                self._reset_context_usage()
                await self._replace_conversation_transcript()
                chat = self.chat()
                if chat is not None:
                    summary = await self.controller.catalog_summary()
                    notice = "Source connection updated. Started a new conversation."
                    if summary.is_empty:
                        notice += " Catalog contains 0 resources; use /source edit to correct its scope."
                    chat.show_notice(notice)
            else:
                await self._show_home_guidance()
            await self._refresh_status()
            return
        if screen_name == "model_setup":
            changed = await self._await_modal(ModelSetupScreen())
            if changed:
                await self.controller.reopen_agent(
                    observer=self._observer,
                    approval_handler=self.handle_approval,
                )
                self._reset_context_usage()
            await self._refresh_status()
            await self._show_home_guidance()
            return
        if screen_name == "permissions":
            await self._await_modal(PermissionsScreen())
            return
        if screen_name == "mcp_management":
            result = await self._await_modal(MCPManagementScreen())
            await self._complete_mcp_screen(result)
            return
        if screen_name == "mcp_setup":
            result = await self._await_modal(MCPSetupScreen())
            await self._complete_mcp_screen(result)
            return
        if screen_name == "jobs":
            await self._await_modal(
                JobsScreen(
                    job_id=(
                        str(payload["job_id"])
                        if isinstance(payload.get("job_id"), str)
                        else None
                    ),
                    initial_view=str(payload.get("view", "details")),
                )
            )
            return
        if screen_name == "routines":
            await self._await_modal(RoutinesScreen())
            await self.refresh_background_status(notify_new=False)
            return
        if screen_name == "inbox":
            await self._await_modal(InboxScreen())
            await self.refresh_background_status(notify_new=False)
            return
        if screen_name == "catalog":
            sources = tuple(
                source
                for source in await self.controller.list_sources()
                if source.active
            )
            resource_groups = await asyncio.gather(
                *(
                    self.controller.list_catalog_resources(source_id=source.id)
                    for source in sources
                )
            )
            resources = tuple(
                resource
                for source_resources in resource_groups
                for resource in source_resources
            )
            current = await self.controller.active_source()
            await self._await_modal(
                CatalogScreen(
                    summary=await self.controller.catalog_summary(),
                    sources=sources,
                    resources=resources,
                    current_source_id=None if current is None else current.id,
                    notice=message,
                    notice_warning=bool(payload.get("catalog_notice_warning", False)),
                )
            )
            if message:
                chat = self.chat()
                if chat is not None:
                    chat.show_notice(message)
            return

        if screen_name == "skill_create":
            await self._create_skill(str(payload.get("name", "")))
            return
        if screen_name == "skill_edit":
            await self._edit_skill(str(payload.get("name", "")))
            return
        if screen_name == "edit_memory":
            await self._edit_memory_target("memory")
            return
        if screen_name == "edit_user":
            await self._edit_memory_target("user")
            return
        if screen_name == "edit_candidate":
            await self._edit_candidate(str(payload.get("candidate_id", "")))
            return
        if screen_name == "review_cost":
            await self._review_with_cost()
            return
        if screen_name == "confirm_clear_conversations":
            accepted = await self._await_modal(ConfirmScreen(message))
            if accepted:
                outcome = await self.controller.clear_conversations()
                self._reset_context_usage()
                await self._replace_conversation_transcript()
                chat = self.chat()
                if chat is not None:
                    chat.show_notice(outcome.message)
            return
        if screen_name == "confirm_delete_agent":
            name = str(payload.get("name", ""))
            accepted = await self._await_modal(
                ConfirmScreen(message, expected_text=name)
            )
            if accepted:
                await self.controller.delete_open_agent()
                self.exit(0)
            return
        if screen_name == "confirm_cancel_job":
            job_id = str(payload.get("job_id", ""))
            accepted = await self._await_modal(ConfirmScreen(message))
            if not accepted:
                return
            inspection = await self.controller.cancel_job(job_id)
            if inspection is None:
                raise UserInputError(
                    "The job no longer exists within this agent boundary."
                )
            status = inspection.summary.status.value
            if status in {"cancel_requested", "cancelled"}:
                notice = "Cancellation requested · " + job_id + " · " + status
            else:
                notice = (
                    "Job became "
                    + status
                    + " before cancellation was applied · "
                    + job_id
                )
            await self._await_modal(
                JobsScreen(
                    job_id=job_id,
                    initial_view="details",
                    notice=notice,
                )
            )
            return
        if screen_name == "confirm_detach_source":
            accepted = await self._await_modal(ConfirmScreen(message))
            if accepted:
                await self.controller.detach_source(str(payload["source_id"]))
                self.controller.conversation_id = None
                self._reset_context_usage()
                await self._replace_conversation_transcript()
            await self._refresh_status()
            return
        if screen_name == "confirm_revoke_mcp":
            accepted = await self._await_modal(ConfirmScreen(message))
            if accepted:
                notice = await self.controller.revoke_mcp_server(
                    str(payload["binding_id"])
                )
                chat = self.chat()
                if chat is not None:
                    chat.show_notice(notice)
            await self._refresh_status()
            return
        if screen_name == "confirm_delete_skill":
            accepted = await self._await_modal(ConfirmScreen(message))
            if accepted:
                name = str(payload["name"])
                deleted = await self.controller.delete_skill(name)
                chat = self.chat()
                if chat is not None:
                    chat.show_notice(
                        f"Skill {name!r} {'deleted' if deleted else 'not found'}."
                    )
            return
        if screen_name == "confirm_delete_semantic":
            accepted = await self._await_modal(ConfirmScreen(message))
            if accepted:
                annotation_id = str(payload["annotation_id"])
                deleted = await self.controller.delete_semantic_annotation(
                    annotation_id
                )
                chat = self.chat()
                if chat is not None:
                    chat.show_notice(
                        f"Semantic annotation {annotation_id!r} "
                        f"{'deleted' if deleted else 'not found'}."
                    )
            return

    async def _complete_mcp_screen(self, result: str | None) -> None:
        chat = self.chat()
        if result == "reopen":
            await self.controller.reopen_agent(
                observer=self._observer,
                approval_handler=self.handle_approval,
            )
            self._reset_context_usage()
            if chat is not None:
                statuses = await self.controller.list_mcp_servers()
                if any(status.reopen_required for status in statuses):
                    chat.show_notice(
                        "The agent runtime restarted, but some MCP tools could not "
                        "be activated. Open /mcp to review their status."
                    )
                else:
                    chat.show_notice("MCP tools activated.")
        elif result == "restart_required" and chat is not None:
            chat.show_notice(
                "MCP changes saved. Restart the agent runtime from /mcp before "
                "using the changed tools."
            )
        await self._refresh_status()

    def _edit_document(self, seed: str) -> str:
        """Give the configured external editor temporary control of the terminal."""

        with self.suspend():
            return self.controller.edit_document(seed)

    async def _edit_memory_target(self, target: str) -> None:
        current = (
            await self.controller.read_memory()
            if target == "memory"
            else await self.controller.read_user_profile()
        )
        edited = self._edit_document(current)
        if target == "memory":
            await self.controller.set_memory(edited)
        else:
            await self.controller.set_user_profile(edited)
        chat = self.chat()
        if chat is not None:
            chat.show_notice(f"{target.capitalize()} updated.")

    async def _edit_skill(self, name: str) -> None:
        skill = await self.controller.read_skill(name)
        if skill is None:
            raise ValueError(f"skill not found: {name}")
        document = render_skill_editor_document(
            skill.name,
            skill.description,
            skill.instructions,
        )
        edited = self._edit_document(document)
        description, instructions = parse_skill_editor_document(name, edited)
        changed = await self.controller.save_skill(name, description, instructions)
        chat = self.chat()
        if chat is not None:
            chat.show_notice(f"Skill {name!r} {'updated' if changed else 'unchanged'}.")

    async def _create_skill(self, requested_name: str) -> None:
        name = requested_name.strip()
        if not name:
            selected = await self._await_modal(SkillNameScreen())
            if selected is None:
                return
            name = selected
        if await self.controller.read_skill(name) is not None:
            raise ValueError(f"skill already exists: {name}")
        seed = render_skill_editor_document(
            name,
            SKILL_DESCRIPTION_PLACEHOLDER,
            SKILL_INSTRUCTIONS_PLACEHOLDER,
        )
        draft = seed
        while True:
            edited = self._edit_document(draft)
            if edited == seed:
                chat = self.chat()
                if chat is not None:
                    chat.show_notice(
                        "Skill creation cancelled; template was unchanged."
                    )
                return
            try:
                description, instructions = parse_skill_editor_document(name, edited)
                if description == SKILL_DESCRIPTION_PLACEHOLDER:
                    raise ValueError("replace the description placeholder")
                if instructions == SKILL_INSTRUCTIONS_PLACEHOLDER:
                    raise ValueError("replace the instructions placeholder")
                if await self.controller.read_skill(name) is not None:
                    raise ValueError(f"skill already exists: {name}")
                changed = await self.controller.save_skill(
                    name,
                    description,
                    instructions,
                )
                if not changed:
                    raise RuntimeError("new skill was not persisted")
            except ValueError as error:
                reopen = await self._await_modal(
                    ConfirmScreen(
                        "Skill document is invalid: "
                        + sanitize_terminal_text(
                            str(error),
                            maximum=320,
                            preserve_lines=False,
                            fallback="invalid document",
                        )
                        + "\nReopen $EDITOR?"
                    )
                )
                if not reopen:
                    return
                draft = edited
                continue
            chat = self.chat()
            if chat is not None:
                chat.show_notice(f"Skill {name!r} created. Invoke it with /{name}.")
            return

    async def _edit_candidate(self, candidate_id: str) -> None:
        document = await self.controller.candidate_editor_document(candidate_id)
        edited = self._edit_document(document)
        await self.controller.save_candidate_document(candidate_id, edited)
        chat = self.chat()
        if chat is not None:
            chat.show_notice(f"Learning candidate {candidate_id!r} updated.")

    async def _review_with_cost(self) -> None:
        value = await self._await_modal(ReviewCostScreen())
        if value is None:
            return
        outcome = await self.controller.review_candidates(value)
        chat = self.chat()
        if chat is not None:
            chat.append_block(
                TranscriptBlock(
                    "notice",
                    f"review-{id(outcome)}",
                    outcome.message,
                )
            )

    async def _start_run(
        self,
        message: str,
        *,
        source_id: str | None = None,
        files_only: bool = False,
        display: str | None = None,
    ) -> None:
        if self._run_task is not None and not self._run_task.done():
            return
        screen = self.chat()
        if screen is None:
            return
        pending_user_identity = f"user-{id(message)}"
        self._pending_user_identity = pending_user_identity
        screen.append_block(
            TranscriptBlock(
                "user",
                pending_user_identity,
                sanitize_terminal_text(
                    display or message,
                    maximum=MAX_COMPOSER_CHARACTERS,
                    preserve_lines=True,
                    fallback="",
                ),
            )
        )
        screen.set_submitting(True)
        if self.controller.conversation_id != self._context_conversation_id:
            self._reset_context_usage()
        screen.set_activity("Thinking", restart=True)
        self._partial_text = ""
        self._run_task = asyncio.create_task(
            self._execute_run(
                message,
                source_id=source_id,
                files_only=files_only,
            ),
            name="daita-agent-run",
        )
        self._run_task.add_done_callback(self._on_run_done)
        await self._refresh_status(running=True, state="thinking")

    async def _execute_run(
        self,
        message: str,
        *,
        source_id: str | None,
        files_only: bool = False,
    ) -> None:
        agent = self.controller.require_agent()
        try:
            result = await agent.run(
                message,
                conversation_id=self.controller.conversation_id,
                source_id=source_id,
                files_only=files_only,
            )
            await self._settle_result(result)
        except asyncio.CancelledError:
            screen = self.chat()
            if screen is not None:
                screen.remove_block(self._partial_identity)
                screen.clear_activity()
                screen.show_notice("Run cancelled.")
            raise
        except Exception as error:
            screen = self.chat()
            if screen is not None:
                screen.remove_block(self._partial_identity)
                screen.clear_activity()
                screen.show_notice(
                    sanitize_terminal_text(
                        str(error),
                        maximum=512,
                        preserve_lines=False,
                        fallback="Run failed.",
                    )
                )
        finally:
            self.invalidate_completion_cache()
            chat = self.chat()
            if chat is not None:
                chat.clear_activity()
                chat.set_submitting(False)
            await self._refresh_status()

    def _on_run_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and not isinstance(error, asyncio.CancelledError):
            return

    async def _settle_result(self, result: LoopExit) -> None:
        self.controller.conversation_id = result.conversation_id
        self._context_conversation_id = result.conversation_id
        screen = self.chat()
        if screen is None:
            return
        screen.remove_block(self._partial_identity)
        if self._pending_user_identity is not None:
            screen.remove_block(self._pending_user_identity)
            self._pending_user_identity = None
        self._partial_text = ""
        await self._replace_conversation_transcript()
        if result.kind is not LoopExitKind.COMPLETED:
            screen.show_notice(_run_failure_notice(result))
        elif result.final_text:
            # Canonical assistant text is already in the transcript.
            pass
        for notice in await self.controller.artifact_notices(result):
            screen.append_block(
                TranscriptBlock(
                    "notice", f"artifact-{result.run_id}-{id(notice)}", notice
                )
            )

    async def on_observer_event(self, message: ObserverEvent) -> None:
        if self._shutting_down:
            return
        event = message.event
        if event.run_origin != "user":
            if event.kind is AgentEventKind.RUN_STARTED:
                self._autonomous_run_ids.add(event.run_id)
                await self._refresh_status()
            elif event.kind is AgentEventKind.RUN_COMPLETED:
                self._autonomous_run_ids.discard(event.run_id)
                await self._refresh_status()
                await self.refresh_background_status(notify_new=True)
            return
        screen = self.chat()
        if screen is None:
            return
        if event.kind is AgentEventKind.MODEL_TEXT_DELTA:
            fragment = event.data.get("text")
            if isinstance(fragment, str) and fragment:
                self._partial_text += fragment
                screen.replace_partial(
                    self._partial_identity,
                    render_model_answer(self._partial_text),
                )
                screen.set_activity("Writing answer")
            return
        if event.kind is AgentEventKind.MODEL_COMPLETED:
            context_input_tokens = event.data.get("context_input_tokens")
            if (
                isinstance(context_input_tokens, int)
                and not isinstance(context_input_tokens, bool)
                and context_input_tokens >= 0
            ):
                self._context_input_tokens = context_input_tokens or None
            await self._refresh_status(running=True, state="working")
            return
        if event.kind is AgentEventKind.TOOL_STARTED:
            raw_tool_name = (
                event.data.get("tool_name") or event.data.get("capability_id") or "tool"
            )
            tool_name = raw_tool_name if isinstance(raw_tool_name, str) else "tool"
            activity = {
                "toolbox_search": "Searching toolboxes",
                "toolbox_load": "Loading selected tools",
                "file_search": "Searching workspace files",
                "file_read": "Reading workspace file",
                "file_query": "Querying workspace data",
                "catalog_search": "Searching catalog",
                "catalog_schema": "Reading catalog schema",
                "data_query_sqlite": "Querying SQLite source",
                "data_query_postgresql": "Querying PostgreSQL source",
                "artifact_create_document": "Creating document",
                "artifact_edit_text": "Preparing workspace file edit",
                "artifact_save_local": "Publishing local artifact",
            }.get(
                tool_name,
                f"Using {CAPABILITY_LABELS.get(tool_name, tool_name)}",
            )
            screen.set_activity(activity)
            await self._refresh_status(running=True, state=activity.casefold())
            return
        if event.kind is AgentEventKind.TOOL_COMPLETED:
            screen.set_activity("Processing results")
            await self._refresh_status(running=True, state="working")
            return
        if event.kind is AgentEventKind.APPROVAL_REQUESTED:
            editing = event.data.get("tool_name") == "artifact_save_local"
            screen.set_activity(
                "Review local file change" if editing else "Waiting for approval"
            )
            await self._refresh_status(
                running=True,
                state="edit approval" if editing else "approval",
            )
            return
        if event.kind is AgentEventKind.APPROVAL_DECIDED:
            screen.set_activity("Applying decision")
            await self._refresh_status(running=True, state="working")
            return
        if event.kind is AgentEventKind.RUN_COMPLETED:
            screen.clear_activity()
            await self._refresh_status(running=False)

    async def copy_or_cancel(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            await self._settle_run()
            return
        screen = self.chat()
        if screen is None:
            return
        text = screen.get_selected_text() or ""
        result = await deliver_clipboard(text)
        screen.show_notice(result.message)

    async def _request_exit(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            accepted = await self._await_modal(
                ConfirmScreen("Cancel the active run and exit?")
            )
            if not accepted:
                return
        self.exit(self._exit_code)

    async def action_quit(self) -> None:
        await self._request_exit()


async def run_daita_app(
    *,
    root: str | Path | None = None,
    workspace: LocalWorkspace,
    agent_name: str | None = None,
    keychain: KeychainStore | None = None,
    model_validator: Any = None,
    reviewer_max_estimated_cost_usd: Decimal | None = None,
) -> int:
    app = DaitaApp(
        root=root,
        workspace=workspace,
        agent_name=agent_name,
        keychain=keychain,
        model_validator=model_validator,
        reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
    )
    result = await app.run_async()
    if app._startup_error is not None:
        raise app._startup_error
    return 0 if result is None else int(result)
