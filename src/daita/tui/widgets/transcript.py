"""Render transcript blocks with follow scrolling and wrap-stable copy text."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from ..models import TranscriptBlock
from ..sanitization import sanitize_markdown, sanitize_terminal_text
from .tool_card import ToolCard


class TranscriptView(VerticalScroll):
    """One vertically scrollable transcript that follows latest until review."""

    def __init__(self) -> None:
        super().__init__(id="transcript")
        self.following = True
        self._blocks: list[TranscriptBlock] = []
        self._tools_visible = False

    def on_scroll(self) -> None:
        if self.max_scroll_y <= 0:
            self.following = True
            return
        self.following = self.scroll_y >= self.max_scroll_y - 1

    @property
    def is_empty(self) -> bool:
        return not self._blocks

    def set_blocks(self, blocks: tuple[TranscriptBlock, ...]) -> None:
        incoming = list(blocks)
        if len(incoming) >= len(self._blocks) and all(
            prior == current
            for prior, current in zip(self._blocks, incoming, strict=False)
        ):
            additions = incoming[len(self._blocks) :]
            self._blocks = incoming
            for block in additions:
                self._mount_block(block)
            self._maybe_follow()
            return
        self._blocks = incoming
        self._rebuild()

    def append_block(self, block: TranscriptBlock) -> None:
        self._blocks.append(block)
        self._mount_block(block)
        self._maybe_follow()

    def replace_partial(self, identity: str, text: str) -> None:
        for index, block in enumerate(self._blocks):
            if block.identity == identity:
                block.text = text
                widget = self.query_one(f"#{_css_id(identity)}", Static)
                widget.update(sanitize_markdown(text))
                self._maybe_follow()
                return
        self.append_block(TranscriptBlock("assistant", identity, text))

    def remove_block(self, identity: str) -> None:
        self._blocks = [block for block in self._blocks if block.identity != identity]
        try:
            self.query_one(f"#{_css_id(identity)}").remove()
        except Exception:
            pass

    def toggle_tools(self) -> None:
        self._tools_visible = not self._tools_visible
        for card in self.query(ToolCard):
            card.display = self._tools_visible
        self._maybe_follow()

    def copy_text(self) -> str:
        parts: list[str] = []
        for block in self._blocks:
            if block.kind == "tool" and not self._tools_visible:
                continue
            if block.kind == "tool" and block.tool_card is not None:
                details = block.tool_card.details
                parts.append(block.tool_card.label)
                if details is not None:
                    parts.append(details.summary)
                continue
            if block.text:
                parts.append(block.text)
        return "\n\n".join(parts)

    def follow_latest(self) -> None:
        self.following = True
        self.scroll_end(animate=False)

    def _rebuild(self) -> None:
        for child in list(self.children):
            child.remove()
        for block in self._blocks:
            self._mount_block(block)
        self._maybe_follow()

    def _mount_block(self, block: TranscriptBlock) -> None:
        if block.kind == "tool":
            if block.tool_card is None:
                return
            card = ToolCard(block.tool_card)
            card.display = self._tools_visible
            self.mount(card)
            return
        classes = "transcript-user" if block.kind == "user" else "transcript-assistant"
        if block.kind == "notice":
            self.mount(
                Static(
                    sanitize_terminal_text(
                        block.text,
                        maximum=4_096,
                        preserve_lines=True,
                        fallback="",
                    ),
                    id=_css_id(block.identity),
                    classes="transcript-notice",
                    markup=False,
                )
            )
            return
        if block.kind == "assistant":
            self.mount(
                Markdown(
                    sanitize_markdown(block.text),
                    id=_css_id(block.identity),
                    classes=classes,
                )
            )
            return
        self.mount(
            Static(
                sanitize_terminal_text(
                    block.text,
                    maximum=16_384,
                    preserve_lines=True,
                    fallback="",
                ),
                id=_css_id(block.identity),
                classes=classes,
                markup=False,
            )
        )

    def _maybe_follow(self) -> None:
        if self.following:
            self.scroll_end(animate=False)


def _css_id(identity: str) -> str:
    return "block-" + "".join(
        character if character.isalnum() else "-" for character in identity
    )
