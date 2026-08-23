"""Browsable current-catalog presentation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Label, Static, Tree

from ..models import SOURCE_TYPE_LABELS
from ..sanitization import safe_display, sanitize_terminal_text


class CatalogScreen(ModalScreen[None]):
    """Group current resources by source in a collapsible tree."""

    BINDINGS = [
        Binding("escape", "close", "Back", priority=True),
        Binding("up", "cursor_up", priority=True, show=False),
        Binding("down", "cursor_down", priority=True, show=False),
        Binding("enter", "toggle_current", priority=True, show=False),
    ]

    def __init__(
        self,
        *,
        summary: Any,
        sources: tuple[Any, ...],
        resources: tuple[Any, ...],
        current_source_id: str | None,
        notice: str = "",
        notice_warning: bool = False,
    ) -> None:
        super().__init__()
        self._summary = summary
        self._sources = sources
        self._resources = resources
        self._current_source_id = current_source_id
        self._notice = notice
        self._notice_warning = notice_warning

    def compose(self) -> ComposeResult:
        with Vertical(id="catalog-browser"):
            yield Label("Catalog", id="catalog-title", markup=False)
            yield Static(self._summary_text(), id="catalog-summary", markup=False)
            if self._notice:
                yield Static(
                    sanitize_terminal_text(
                        self._notice,
                        maximum=512,
                        preserve_lines=False,
                        fallback="Catalog refresh succeeded.",
                    ),
                    id="catalog-notice",
                    classes="-warning" if self._notice_warning else "",
                    markup=False,
                )
            tree: Tree[str] = Tree("Sources", id="catalog-tree")
            tree.show_root = False
            yield tree
            yield Static(
                "Click a source to expand/collapse  ·  ↑/↓ select  ·  Enter toggle",
                id="catalog-help",
                markup=False,
            )
            yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#catalog-tree", Tree)
        resources_by_source: dict[str, list[Any]] = defaultdict(list)
        for resource in self._resources:
            resources_by_source[resource.source_id].append(resource)

        ordered_sources = sorted(
            self._sources,
            key=lambda source: (
                source.id != self._current_source_id,
                source.display_name.casefold(),
                source.id,
            ),
        )
        for source in ordered_sources:
            source_resources = sorted(
                resources_by_source.get(source.id, ()),
                key=lambda resource: (
                    resource.native_identity.casefold(),
                    resource.id,
                ),
            )
            source_node = tree.root.add(
                self._source_label(source, len(source_resources)), data=source.id
            )
            for resource in source_resources:
                source_node.add_leaf(self._resource_label(resource), data=resource.id)
            if not source_resources:
                source_node.add_leaf(Text("No current resources", style="dim"))
            source_node.expand()

        tree.root.expand()
        if ordered_sources:
            tree.cursor_line = 0
        tree.focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        self.query_one("#catalog-tree", Tree).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#catalog-tree", Tree).action_cursor_down()

    def action_toggle_current(self) -> None:
        self.query_one("#catalog-tree", Tree).action_toggle_node()

    def _summary_text(self) -> str:
        return (
            f"{self._summary.active_source_count} sources  ·  "
            f"{self._summary.resource_count} resources  ·  "
            f"{self._summary.relationship_count} relationships"
        )

    def _source_label(self, source: Any, resource_count: int) -> Text:
        label = Text()
        if source.id == self._current_source_id:
            label.append("● ", style="#ACFD21")
        label.append(
            safe_display(source.display_name, fallback="source", maximum=512),
            style="bold",
        )
        source_type = SOURCE_TYPE_LABELS.get(source.adapter_id, source.adapter_id)
        noun = "resource" if resource_count == 1 else "resources"
        label.append(f"  {source_type} · {resource_count} {noun}", style="dim")
        if source.id == self._current_source_id:
            label.append("  current", style="#ACFD21")
        return label

    @staticmethod
    def _resource_label(resource: Any) -> Text:
        label = Text(
            safe_display(
                resource.native_identity,
                fallback=resource.name,
                maximum=512,
            )
        )
        kind = getattr(resource.kind, "value", str(resource.kind))
        label.append(f"  {kind}", style="dim")
        return label
