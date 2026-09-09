"""Bounded receipt inspection and exact human recovery through Agent controls."""

from __future__ import annotations

import json

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita import (
    ApprovalDecision,
    ApprovalRequest,
    EffectReceipt,
    EffectResolutionDecision,
)

from ..projection import effect_receipt_mapping
from ..sanitization import safe_display
from ..widgets.approval import ApprovalPanel


class EffectsScreen(ModalScreen[None]):
    """A human decision never dispatches or reconciles the original operation."""

    BINDINGS = [Binding("escape", "close", "Back")]

    def __init__(self, *, receipt_id: str | None = None) -> None:
        super().__init__()
        self._target_id = receipt_id
        self._receipts: tuple[EffectReceipt, ...] = ()
        self._receipt: EffectReceipt | None = None
        self._offset = 0
        self._unresolved_only = True
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="effects-manager"):
            yield Label("Action receipts and human recovery", markup=False)
            yield Static(
                "Host open here. Recovery records a decision and performs no action.",
                markup=False,
            )
            yield OptionList(id="effects-list")
            with VerticalScroll(id="effects-detail-scroll"):
                yield Static("", id="effects-detail", markup=False)
            yield Input(placeholder="Required investigation note", id="effects-note")
            yield Input(
                placeholder="Optional exact receipt/artifact evidence IDs, separated by spaces",
                id="effects-evidence",
            )
            with Horizontal(classes="effects-actions"):
                yield Button("Refresh", id="effects-refresh")
                yield Button("Show all", id="effects-filter")
                yield Button("Previous", id="effects-previous")
                yield Button("Next", id="effects-next")
            with Horizontal(classes="effects-actions"):
                yield Button("Close without retry", id="effects-close-without-retry")
                yield Button("Allow future work", id="effects-allow-future-work")
                yield Button("Back", id="effects-back")
            yield Static("", id="effects-error", markup=False)
            yield ApprovalPanel()
            yield Footer()

    def on_mount(self) -> None:
        self._schedule("refresh")

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        identity = event.button.id or ""
        if not identity.startswith("effects-"):
            return
        if identity == "effects-back":
            self.action_close()
        else:
            self._schedule(identity.removeprefix("effects-"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        del event
        self._schedule("inspect")

    def _schedule(self, action: str) -> None:
        if not self._busy:
            self.run_worker(
                self._handle(action), group="effects-interaction", exclusive=True
            )

    async def request_approval(
        self, request: ApprovalRequest
    ) -> ApprovalDecision | None:
        controls = tuple(
            self.query(
                "#effects-list, #effects-detail-scroll, #effects-note, #effects-evidence, .effects-actions, #effects-error"
            )
        )
        for control in controls:
            control.display = False
        try:
            return await self.query_one(ApprovalPanel).request(request)
        finally:
            for control in controls:
                control.display = True

    async def _handle(self, action: str) -> None:
        self._busy = True
        self.query_one("#effects-error", Static).update("")
        try:
            controller = self.app.controller  # type: ignore[attr-defined]
            if action in {"close-without-retry", "allow-future-work"}:
                receipt = self._receipt
                if receipt is None:
                    raise ValueError("Inspect an exact receipt first.")
                await controller.resolve_effect(
                    receipt.receipt_id,
                    expected_digest=receipt.receipt_digest,
                    decision=EffectResolutionDecision(action.replace("-", "_")),
                    note=self.query_one("#effects-note", Input).value,
                    evidence_references=tuple(
                        self.query_one("#effects-evidence", Input).value.split()
                    ),
                )
                self._target_id = receipt.receipt_id
                self.query_one("#effects-error", Static).update(
                    "Recovery recorded. No action was performed. Allow future work leaves the routine paused; "
                    "close without retry disables it. Review /routines separately."
                )
            elif action == "filter":
                self._unresolved_only = not self._unresolved_only
                self._offset = 0
            elif action == "next":
                self._offset += 20
            elif action == "previous":
                self._offset = max(0, self._offset - 20)
            elif action == "inspect":
                index = self.query_one("#effects-list", OptionList).highlighted
                if index is None or not 0 <= index < len(self._receipts):
                    raise ValueError("Select a receipt first.")
                self._target_id = self._receipts[index].receipt_id
            if action != "inspect":
                self._receipts = await controller.list_effects(
                    unresolved_only=self._unresolved_only,
                    limit=20,
                    offset=self._offset,
                )
                listing = self.query_one("#effects-list", OptionList)
                listing.clear_options()
                listing.add_options(
                    [
                        Option(
                            Text(
                                f"{item.receipt_id} · {item.outcome.value} · {item.evidence_basis.value}"
                            ),
                            id=item.receipt_id,
                        )
                        for item in self._receipts
                    ]
                )
                self.query_one("#effects-filter", Button).label = (
                    "Show all" if self._unresolved_only else "Unresolved only"
                )
            if self._target_id is None and self._receipts:
                self._target_id = self._receipts[0].receipt_id
            if self._target_id is not None:
                self._receipt = await controller.inspect_effect(self._target_id)
                if self._receipt is None:
                    raise ValueError("No receipt with that ID belongs to this agent.")
                receipt = self._receipt
                self.query_one("#effects-detail", Static).update(
                    Text(
                        "Adapter-verified evidence describes the native transaction. Server-reported evidence "
                        "describes invocation only. Uncertain means the action may have happened.\n\n"
                        + json.dumps(
                            effect_receipt_mapping(receipt),
                            ensure_ascii=True,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                )
            else:
                self.query_one("#effects-detail", Static).update(
                    "No receipts in this page."
                )
        except (ValueError, RuntimeError, PermissionError, OSError) as error:
            self.query_one("#effects-error", Static).update(
                safe_display(str(error), maximum=2048)
            )
        finally:
            self._busy = False
            receipt = self._receipt
            unresolved = (
                receipt is not None
                and receipt.outcome.value == "uncertain"
                and receipt.resolution is None
            )
            for identity in ("close-without-retry", "allow-future-work"):
                self.query_one(f"#effects-{identity}", Button).disabled = not unresolved
            self.query_one("#effects-previous", Button).disabled = self._offset == 0
            self.query_one("#effects-next", Button).disabled = len(self._receipts) < 20
