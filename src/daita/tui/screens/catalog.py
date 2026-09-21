"""Display searchable and browsable views of the current catalog."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, ItemGrid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, Static, Tree

from ..models import SOURCE_TYPE_LABELS
from ..sanitization import safe_display, sanitize_terminal_text


@dataclass(frozen=True)
class SourceManagerAction:
    """One source-management action selected from the sources screen."""

    kind: str
    source_id: str | None = None


@dataclass(frozen=True)
class CatalogGraphNode:
    """Presentation-only identity for one node in the relationship tree."""

    kind: str
    resource_id: str
    relationship_id: str | None = None


class CatalogScreen(ModalScreen[SourceManagerAction | None]):
    """Browse source catalogs and choose contextual source-management actions."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Back", priority=True),
        Binding("up", "cursor_up", priority=True, show=False),
        Binding("down", "cursor_down", priority=True, show=False),
        Binding("enter", "toggle_current", priority=True, show=False),
        Binding("s", "show_sources", "Sources", priority=True),
        Binding("g", "show_graph", "Tree graph", priority=True),
        Binding("backspace", "graph_back", priority=True, show=False),
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
        self._resources_by_id = {resource.id: resource for resource in resources}
        self._ordered_resources = tuple(
            sorted(
                resources,
                key=lambda resource: (
                    resource.native_identity.casefold(),
                    resource.source_id,
                    resource.id,
                ),
            )
        )
        self._mode = "sources"
        self._graph_root_id: str | None = None
        self._graph_history: list[str] = []
        self._graph_resources: dict[str, Mapping[str, object]] = {}
        self._graph_relationships: dict[str, Mapping[str, object]] = {}
        self._graph_busy = False

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
            with Horizontal(id="catalog-modes"):
                yield Button(
                    "Sources",
                    id="catalog-mode-sources",
                    variant="primary",
                )
                yield Button("Tree graph", id="catalog-mode-graph")
            yield self._catalog_tree()
            with Horizontal(id="catalog-graph-view"):
                yield Tree[CatalogGraphNode](
                    "Select a resource",
                    id="catalog-graph-tree",
                )
                yield Static(
                    "Select Tree graph to explore catalog relationships.",
                    id="catalog-graph-detail",
                    markup=False,
                )
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
            with Horizontal(id="catalog-graph-actions"):
                yield Button("Back", id="catalog-graph-back")
                yield Button(
                    "Open selected",
                    id="catalog-graph-open",
                    variant="primary",
                )
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
        graph = self.query_one("#catalog-graph-tree", Tree)
        graph.show_root = True
        self._set_mode("sources")
        self._apply_responsive_layout()

    def on_resize(self) -> None:
        self._apply_responsive_layout()

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
        if button_id == "catalog-mode-sources":
            self.action_show_sources()
            return
        if button_id == "catalog-mode-graph":
            self.action_show_graph()
            return
        if button_id == "catalog-graph-back":
            self.action_graph_back()
            return
        if button_id == "catalog-graph-open":
            self._open_selected_graph_neighbor()
            return
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

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        if event.control.id == "catalog-tree":
            self._refresh_actions()
            return
        if event.control.id == "catalog-graph-tree":
            self._render_graph_detail(event.node.data)
            self._refresh_graph_actions()

    def action_cursor_up(self) -> None:
        self._active_tree().action_cursor_up()

    def action_cursor_down(self) -> None:
        self._active_tree().action_cursor_down()

    def action_toggle_current(self) -> None:
        if self._mode == "sources":
            self.query_one("#catalog-tree", Tree).action_toggle_node()
            return
        selected = self._selected_graph_node()
        if selected is not None and selected.kind == "neighbor":
            self._schedule_graph_load(selected.resource_id, push_history=True)
            return
        self.query_one("#catalog-graph-tree", Tree).action_toggle_node()

    def action_show_sources(self) -> None:
        self._set_mode("sources")

    def action_show_graph(self) -> None:
        if not self._ordered_resources:
            self._set_mode("graph")
            self._show_graph_empty_state()
            return
        selected = self._selected_resource_id()
        resource_id = selected or self._graph_root_id or self._ordered_resources[0].id
        self._set_mode("graph")
        if resource_id != self._graph_root_id or not self._graph_resources:
            if self._graph_root_id is not None and resource_id != self._graph_root_id:
                self._graph_history.clear()
            self._schedule_graph_load(resource_id, push_history=False)

    def action_graph_back(self) -> None:
        if self._mode != "graph" or self._graph_busy or not self._graph_history:
            return
        self._schedule_graph_load(
            self._graph_history[-1],
            push_history=False,
            pop_history=True,
        )

    def _active_tree(self) -> Tree[Any]:
        tree_id = "#catalog-tree" if self._mode == "sources" else "#catalog-graph-tree"
        return self.query_one(tree_id, Tree)

    def _set_mode(self, mode: str) -> None:
        if mode not in {"sources", "graph"}:
            raise ValueError("unknown catalog display mode")
        self._mode = mode
        sources_visible = mode == "sources"
        self.query_one("#catalog-tree", Tree).display = sources_visible
        self.query_one("#source-actions", ItemGrid).display = sources_visible
        self.query_one("#catalog-graph-view", Horizontal).display = not sources_visible
        self.query_one("#catalog-graph-actions", Horizontal).display = (
            not sources_visible
        )
        self.query_one("#catalog-mode-sources", Button).variant = (
            "primary" if sources_visible else "default"
        )
        self.query_one("#catalog-mode-graph", Button).variant = (
            "default" if sources_visible else "primary"
        )
        self.query_one("#catalog-title", Label).update(
            "Sources" if sources_visible else "Catalog tree graph"
        )
        if sources_visible:
            self.query_one("#catalog-tree", Tree).focus()
            self._refresh_actions()
        else:
            self.query_one("#catalog-graph-tree", Tree).focus()
            self._refresh_graph_actions()

    def _selected_resource_id(self) -> str | None:
        node = self.query_one("#catalog-tree", Tree).cursor_node
        if node is None:
            return None
        if node.data in self._resources_by_id:
            return node.data
        source_id = self._selected_source_id()
        centered = self._resources_by_id.get(self._graph_root_id or "")
        if centered is not None and centered.source_id == source_id:
            return centered.id
        return next(
            (
                resource.id
                for resource in self._ordered_resources
                if resource.source_id == source_id
            ),
            None,
        )

    def _selected_graph_node(self) -> CatalogGraphNode | None:
        node = self.query_one("#catalog-graph-tree", Tree).cursor_node
        return node.data if node is not None else None

    def _open_selected_graph_neighbor(self) -> None:
        selected = self._selected_graph_node()
        if selected is None or selected.kind != "neighbor":
            return
        self._schedule_graph_load(selected.resource_id, push_history=True)

    def _schedule_graph_load(
        self,
        resource_id: str,
        *,
        push_history: bool,
        pop_history: bool = False,
    ) -> None:
        if self._graph_busy or resource_id not in self._resources_by_id:
            return
        self.run_worker(
            self._load_graph_resource(
                resource_id,
                push_history=push_history,
                pop_history=pop_history,
            ),
            name="catalog-tree-graph-load",
            group="catalog-tree-graph",
            exclusive=True,
        )

    async def _load_graph_resource(
        self,
        resource_id: str,
        *,
        push_history: bool,
        pop_history: bool,
    ) -> None:
        previous_root_id = self._graph_root_id
        self._set_graph_busy(True)
        self.query_one("#catalog-graph-detail", Static).update(
            "Loading the selected resource neighborhood…"
        )
        try:
            frozen = await self.app.controller.inspect_catalog_resource(  # type: ignore[attr-defined]
                resource_id
            )
            inspection = frozen.to_dict()
            self._render_graph_tree(resource_id, inspection)
        except (TypeError, ValueError, RuntimeError, OSError) as error:
            message = sanitize_terminal_text(
                str(error),
                maximum=512,
                preserve_lines=False,
                fallback="The catalog neighborhood could not be loaded.",
            )
            self.query_one("#catalog-graph-detail", Static).update(
                "Catalog graph unavailable\n\n" + message
            )
            return
        finally:
            if self.is_mounted:
                self._set_graph_busy(False)

        if pop_history and self._graph_history:
            self._graph_history.pop()
        elif (
            push_history
            and previous_root_id is not None
            and previous_root_id != resource_id
        ):
            self._graph_history.append(previous_root_id)
        self._graph_root_id = resource_id
        self._select_source_tree_resource(resource_id)
        self._refresh_graph_actions()

    def _select_source_tree_resource(self, resource_id: str) -> None:
        tree = self.query_one("#catalog-tree", Tree)
        node = next(
            (
                resource_node
                for source_node in tree.root.children
                for resource_node in source_node.children
                if resource_node.data == resource_id
            ),
            None,
        )
        if node is not None:
            tree.move_cursor(node)

    def _render_graph_tree(
        self,
        resource_id: str,
        inspection: Mapping[str, object],
    ) -> None:
        resource_payload = _mapping(inspection.get("resource"))
        relationships = tuple(
            _mapping(item)
            for item in _sequence(inspection.get("incident_relationships"))
        )
        neighbors = tuple(
            _mapping(item) for item in _sequence(inspection.get("neighbors"))
        )
        if resource_payload.get("resource_id") != resource_id:
            raise ValueError("catalog inspection returned another resource")

        self._graph_resources = {
            str(item["resource_id"]): item
            for item in (resource_payload, *neighbors)
            if isinstance(item.get("resource_id"), str)
        }
        self._graph_relationships = {
            str(item["relationship_id"]): item
            for item in relationships
            if isinstance(item.get("relationship_id"), str)
        }

        tree = self.query_one("#catalog-graph-tree", Tree)
        tree.reset(
            self._graph_resource_label(resource_id, centered=True),
            CatalogGraphNode("resource", resource_id),
        )
        ordered_relationships = sorted(
            relationships,
            key=lambda relationship: self._relationship_sort_key(
                resource_id, relationship
            ),
        )
        for relationship in ordered_relationships:
            relationship_id = str(relationship["relationship_id"])
            neighbor_id = self._relationship_neighbor_id(resource_id, relationship)
            relation_node = tree.root.add(
                self._graph_relationship_label(relationship),
                CatalogGraphNode(
                    "relationship",
                    resource_id,
                    relationship_id,
                ),
            )
            relation_node.add_leaf(
                self._graph_resource_label(neighbor_id),
                CatalogGraphNode("neighbor", neighbor_id, relationship_id),
            )
            relation_node.expand()
        if not ordered_relationships:
            tree.root.add_leaf(
                Text("No catalog relationships", style="dim"),
                CatalogGraphNode("empty", resource_id),
            )
        tree.root.expand()
        tree.move_cursor(tree.root)
        self._render_graph_detail(tree.root.data)

        if bool(inspection.get("incident_relationships_truncated", False)):
            self.query_one("#catalog-help", Static).update(
                "Showing the bounded relationship neighborhood; additional edges exist. "
                "Select a neighbor and press Enter to continue exploring."
            )
        else:
            self.query_one("#catalog-help", Static).update(
                "Up/Down select · Enter or Open selected follows a neighbor · "
                "Backspace returns · catalog metadata does not grant execution authority"
            )

    def _render_graph_detail(self, selected: CatalogGraphNode | None) -> None:
        detail = self.query_one("#catalog-graph-detail", Static)
        if selected is None:
            detail.update("Select a catalog graph node to inspect it.")
            return
        if selected.kind == "relationship" and selected.relationship_id is not None:
            relationship = self._graph_relationships.get(selected.relationship_id)
            if relationship is None:
                detail.update("Relationship details are unavailable.")
                return
            from_id = str(relationship.get("from_resource_id", ""))
            to_id = str(relationship.get("to_resource_id", ""))
            fields = tuple(
                _mapping(item) for item in _sequence(relationship.get("field_pairs"))
            )
            lines = [
                "Relationship",
                "",
                "Kind: " + _display_value(relationship.get("kind"), "unknown"),
                "Direction from center: "
                + (
                    "outgoing"
                    if relationship.get("direction") == "forward"
                    else "incoming"
                ),
                "From: " + self._resource_name(from_id),
                "To: " + self._resource_name(to_id),
                "Provenance: "
                + _display_value(relationship.get("provenance"), "unknown"),
                "Confidence: " + _confidence_text(relationship.get("confidence")),
            ]
            if fields:
                lines.extend(("", "Fields"))
                lines.extend(
                    "  "
                    + _display_value(pair.get("source_field"), "?")
                    + " → "
                    + _display_value(pair.get("target_field"), "?")
                    for pair in fields
                )
            lines.extend(
                (
                    "",
                    (
                        "Catalog evidence describes structure; it does not grant "
                        "access or execution authority."
                    ),
                )
            )
            detail.update("\n".join(lines))
            return

        resource = self._resources_by_id.get(selected.resource_id)
        payload = self._graph_resources.get(selected.resource_id, {})
        if resource is None:
            detail.update("Resource details are unavailable.")
            return
        source = self._sources_by_id.get(resource.source_id)
        source_name = (
            safe_display(source.display_name, fallback="source")
            if source is not None
            else "Unknown source"
        )
        kind = getattr(resource.kind, "value", str(resource.kind))
        lines = [
            safe_display(resource.native_identity, fallback=resource.name),
            "",
            "Kind: " + safe_display(kind, fallback="resource"),
            "Source: " + source_name,
        ]
        sensitivity = payload.get("sensitivity")
        if sensitivity is not None:
            lines.append("Sensitivity: " + _display_value(sensitivity, "unknown"))
        if selected.kind == "resource":
            lines.append(
                f"Relationships: {len(self._graph_relationships)} in this neighborhood"
            )
        elif selected.kind == "neighbor":
            lines.extend(("", "Press Enter or choose Open selected to center here."))
        detail.update("\n".join(lines))

    def _relationship_sort_key(
        self,
        resource_id: str,
        relationship: Mapping[str, object],
    ) -> tuple[str, str, str]:
        neighbor_id = self._relationship_neighbor_id(resource_id, relationship)
        return (
            _display_value(relationship.get("kind"), "").casefold(),
            self._resource_name(neighbor_id).casefold(),
            _display_value(relationship.get("relationship_id"), ""),
        )

    @staticmethod
    def _relationship_neighbor_id(
        resource_id: str,
        relationship: Mapping[str, object],
    ) -> str:
        from_id = relationship.get("from_resource_id")
        to_id = relationship.get("to_resource_id")
        if from_id == resource_id and isinstance(to_id, str):
            return to_id
        if to_id == resource_id and isinstance(from_id, str):
            return from_id
        raise ValueError("catalog relationship does not include the centered resource")

    def _graph_relationship_label(self, relationship: Mapping[str, object]) -> Text:
        kind = _display_value(relationship.get("kind"), "relationship")
        forward = relationship.get("direction") == "forward"
        label = Text(f"{kind} →" if forward else f"← {kind}")
        label.append(
            "  " + _display_value(relationship.get("provenance"), "unknown"),
            style="dim",
        )
        return label

    def _graph_resource_label(
        self, resource_id: str, *, centered: bool = False
    ) -> Text:
        label = Text(self._resource_name(resource_id), style="bold" if centered else "")
        resource = self._resources_by_id.get(resource_id)
        if resource is not None:
            kind = getattr(resource.kind, "value", str(resource.kind))
            label.append(f"  {kind}", style="dim")
        return label

    def _resource_name(self, resource_id: str) -> str:
        resource = self._resources_by_id.get(resource_id)
        if resource is not None:
            return safe_display(
                resource.native_identity,
                fallback=resource.name,
                maximum=512,
            )
        payload = self._graph_resources.get(resource_id, {})
        return _display_value(payload.get("name"), "Unknown resource")

    def _set_graph_busy(self, busy: bool) -> None:
        self._graph_busy = busy
        self.query_one("#catalog-graph-tree", Tree).disabled = busy
        self._refresh_graph_actions()

    def _refresh_graph_actions(self) -> None:
        if not self.is_mounted:
            return
        selected = self._selected_graph_node()
        self.query_one("#catalog-graph-back", Button).disabled = (
            self._graph_busy or not self._graph_history
        )
        self.query_one("#catalog-graph-open", Button).disabled = (
            self._graph_busy or selected is None or selected.kind != "neighbor"
        )

    def _show_graph_empty_state(self) -> None:
        tree = self.query_one("#catalog-graph-tree", Tree)
        tree.reset("Catalog graph", CatalogGraphNode("empty", ""))
        tree.root.add_leaf(Text("No current resources", style="dim"))
        tree.root.expand()
        self.query_one("#catalog-graph-detail", Static).update(
            "No resources are available. Attach or refresh a source first."
        )
        self.query_one("#catalog-help", Static).update(
            "The tree graph is read-only catalog evidence."
        )
        self._refresh_graph_actions()

    def _apply_responsive_layout(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#catalog-graph-view", Horizontal).set_class(
            self.size.width < 82,
            "-compact",
        )

    def _selected_source_id(self) -> str | None:
        node = self.query_one("#catalog-tree", Tree).cursor_node
        while node is not None and node.parent is not None:
            if node.data in self._source_ids:
                return node.data
            node = node.parent
        return None

    def _refresh_actions(self) -> None:
        if not self.is_mounted or self._mode != "sources":
            return
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


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("catalog inspection contained a non-object value")
    return value


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("catalog inspection contained a non-list value")
    return tuple(value)


def _display_value(value: object, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    return safe_display(value, fallback=fallback, maximum=512)


def _confidence_text(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unknown"
    return f"{float(value):.0%}"
