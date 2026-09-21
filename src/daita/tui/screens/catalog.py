"""Display searchable and browsable views of the current catalog."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ItemGrid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, Static, Tree

from ..models import SOURCE_TYPE_LABELS
from ..sanitization import safe_display, sanitize_terminal_text


@dataclass(frozen=True)
class SourceManagerAction:
    """One source-management action selected from the sources screen."""

    kind: str
    source_id: str | None = None


class CatalogScreen(ModalScreen[SourceManagerAction | None]):
    """Browse source catalogs and choose contextual source-management actions."""

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
        notice: str = "",
        notice_warning: bool = False,
        initial_source_id: str | None = None,
    ) -> None:
        super().__init__()
        self._summary = summary
        self._sources = sources
        self._resources = resources
        self._notice = notice
        self._notice_warning = notice_warning
        self._initial_source_id = initial_source_id
        self._source_ids = frozenset(source.id for source in sources)
        self._sources_by_id = {source.id: source for source in sources}

    def compose(self) -> ComposeResult:
        with Vertical(id="catalog-browser"):
            yield Label("Sources", id="catalog-title", markup=False)
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
            yield self._catalog_tree()
            yield Static(
                "Select a source or one of its resources to manage it.",
                id="catalog-help",
                markup=False,
            )
            with ItemGrid(id="source-actions", min_column_width=14):
                yield Button("Add source", id="source-add", variant="primary")
                yield Button("Refresh", id="source-refresh")
                yield Button("Edit", id="source-edit")
                yield Button("Permissions", id="source-permissions")
                yield Button("Detach", id="source-detach", variant="error")
            yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#catalog-tree", Tree)
        if self._initial_source_id is not None:
            initial_node = next(
                (
                    node
                    for node in tree.root.children
                    if node.data == self._initial_source_id
                ),
                None,
            )
            if initial_node is not None:
                tree.move_cursor(initial_node)
        tree.focus()
        self._refresh_actions()

    def _catalog_tree(self) -> Tree[str]:
        # Publish the notice and its source/resource contents in one composition;
        # another task can inspect the screen before its Mount message runs.
        tree: Tree[str] = Tree("Sources", id="catalog-tree")
        tree.show_root = False
        resources_by_source: dict[str, list[Any]] = defaultdict(list)
        for resource in self._resources:
            resources_by_source[resource.source_id].append(resource)

        ordered_sources = sorted(
            self._sources,
            key=lambda source: (
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
        return tree

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "source-add":
            self.dismiss(SourceManagerAction("add"))
            return
        actions = {
            "source-refresh": "refresh",
            "source-edit": "edit",
            "source-permissions": "permissions",
            "source-detach": "detach",
        }
        if button_id not in actions:
            return
        source_id = self._selected_source_id()
        if source_id is not None:
            self.dismiss(SourceManagerAction(actions[button_id], source_id))

    def on_tree_node_highlighted(self, _event: Tree.NodeHighlighted[str]) -> None:
        self._refresh_actions()

    def action_cursor_up(self) -> None:
        self.query_one("#catalog-tree", Tree).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#catalog-tree", Tree).action_cursor_down()

    def action_toggle_current(self) -> None:
        self.query_one("#catalog-tree", Tree).action_toggle_node()

    def _selected_source_id(self) -> str | None:
        node = self.query_one("#catalog-tree", Tree).cursor_node
        while node is not None and node.parent is not None:
            if node.data in self._source_ids:
                return node.data
            node = node.parent
        return None

    def _refresh_actions(self) -> None:
        source_id = self._selected_source_id()
        for button_id in (
            "source-refresh",
            "source-edit",
            "source-permissions",
            "source-detach",
        ):
            self.query_one(f"#{button_id}", Button).disabled = source_id is None
        if source_id is None:
            message = (
                "No sources are attached. Add one to make data available."
                if not self._sources
                else "Select a source or one of its resources to manage it."
            )
        else:
            source = self._sources_by_id[source_id]
            message = (
                safe_display(source.display_name, fallback="Selected source")
                + " selected  ·  Enter expands or collapses its resources"
            )
        self.query_one("#catalog-help", Static).update(message)

    def _summary_text(self) -> str:
        return (
            f"{self._summary.active_source_count} sources  ·  "
            f"{self._summary.resource_count} resources  ·  "
            f"{self._summary.relationship_count} relationships"
        )

    def _source_label(self, source: Any, resource_count: int) -> Text:
        label = Text()
        label.append(
            safe_display(source.display_name, fallback="source", maximum=512),
            style="bold",
        )
        source_type = SOURCE_TYPE_LABELS.get(source.adapter_id, source.adapter_id)
        noun = "resource" if resource_count == 1 else "resources"
        label.append(f"  {source_type} · {resource_count} {noun}", style="dim")
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
