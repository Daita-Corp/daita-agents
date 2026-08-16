"""Bounded tool status and result presentation."""

from __future__ import annotations

from textual.widgets import Collapsible, DataTable, Static

from ..models import ToolCardState
from ..sanitization import sanitize_terminal_text


class ToolCard(Static):
    def __init__(self, card: ToolCardState) -> None:
        super().__init__(id=f"tool-{card.call_id}")
        self.card = card

    def compose(self):
        title = sanitize_terminal_text(
            f"{self.card.label} · {self.card.state}",
            maximum=240,
            preserve_lines=False,
            fallback="tool",
        )
        with Collapsible(title=title, collapsed=not self.card.expanded):
            yield Static(
                self._detail_text(), markup=False, id=f"tool-detail-{self.card.call_id}"
            )
            if self.card.details is not None and self.card.details.table is not None:
                yield self._table(self.card.details.table)

    def _detail_text(self) -> str:
        details = self.card.details
        if details is None:
            return sanitize_terminal_text(
                self.card.state,
                maximum=240,
                preserve_lines=False,
                fallback="working",
            )
        parts = [details.summary]
        if details.error_message:
            parts.append(details.error_message)
        if details.code:
            parts.append(details.code)
        if details.arguments_text:
            parts.append(details.arguments_text)
        if details.result_text:
            parts.append(details.result_text)
        return "\n\n".join(parts)

    def _table(self, preview) -> DataTable[str]:
        table: DataTable[str] = DataTable(id=f"tool-table-{self.card.call_id}")
        table.add_columns(*preview.columns)
        for row in preview.rows[:10]:
            table.add_row(*row)
        return table

    def refresh_card(self, card: ToolCardState) -> None:
        self.card = card
        detail = self.query_one(f"#tool-detail-{card.call_id}", Static)
        detail.update(self._detail_text())
