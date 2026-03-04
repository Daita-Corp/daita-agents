"""
ConversationHistory — Lightweight persistent conversation state for Agent.

Allows any Agent to function as a stateful chatbot across multiple run() calls.
History is just a list of message dicts persisted to .daita/sessions/.

Usage:
    history = ConversationHistory(session_id="alice", workspace="support_bot")
    response = await agent.run("Hi, I'm Alice.", history=history)
    response = await agent.run("What's my name?", history=history)  # knows it's Alice
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles

logger = logging.getLogger(__name__)


def _find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    current = start_path or Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "daita-project.yaml").exists():
            return parent
    return None


def _derive_workspace(agent_id: str) -> str:
    """Strip UUID suffix from agent_id: 'support_bot_a1b2c3ef' → 'support_bot'."""
    if '_' in agent_id and len(agent_id) > 9:
        return '_'.join(agent_id.split('_')[:-1])
    return agent_id


def _estimate_tokens(messages: List[Dict]) -> int:
    """Rough estimate: 4 characters ≈ 1 token (suitable for English text)."""
    return sum(len(m.get("content", "")) for m in messages) // 4


class ConversationHistory:
    """
    Lightweight conversation history value object.

    Stores the sequence of user/assistant message pairs for a session and
    can persist them to .daita/sessions/{workspace}/{session_id}.json.

    Not a plugin. Not a subclass of BaseAgent. Agent stays stateless —
    history lives here, not on the agent.
    """

    def __init__(
        self,
        session_id: str,
        workspace: Optional[str] = None,
        max_turns: Optional[int] = None,
        max_tokens: Optional[int] = None,
        auto_save: bool = False,
        scope: str = "project",
        base_dir: Optional[Path] = None,
    ):
        """
        Create a ConversationHistory.

        Args:
            session_id: Unique identifier for this conversation session.
            workspace: Storage namespace, usually the agent name. If omitted,
                the agent derives it automatically on the first run() call.
                Required for load() and save() without a prior run().
            max_turns: Sliding window — keep only the last N complete
                user+assistant pairs. None means unlimited.
            max_tokens: Token budget for history. Oldest turns are dropped when
                the estimated token count exceeds this limit. Applied after
                max_turns. Recommended values: 3,000 (GPT-4/8k),
                50,000 (GPT-4o/128k), 100,000 (Claude/200k).
            auto_save: When True, save() is called automatically after every
                add_turn(). Recommended for production chatbots.
            scope: "project" (default) stores under the nearest
                daita-project.yaml; "global" stores under ~/.daita/.
            base_dir: Override path resolution entirely — useful in tests.
        """
        self.session_id = session_id
        self.workspace = workspace
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.auto_save = auto_save
        self._scope = scope
        self._base_dir = base_dir
        self._messages: List[Dict] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def messages(self) -> List[Dict]:
        """Read-only snapshot of current history as a list of message dicts."""
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Number of complete user+assistant turns currently stored."""
        return len(self._messages) // 2

    @property
    def session_path(self) -> Optional[Path]:
        """Resolved file path, or None if workspace is not yet set."""
        if self.workspace is None:
            return None
        return self._resolve_path()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def add_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Append a completed turn. Called by the agent automatically after run().

        Applies max_turns then max_tokens windowing, then auto-saves if
        auto_save=True. Should not normally be called directly by developers.
        """
        self._messages.append({"role": "user", "content": user_message})
        self._messages.append({"role": "assistant", "content": assistant_message})
        if self.max_turns is not None:
            self._messages = self._messages[-(self.max_turns * 2):]
        if self.max_tokens is not None:
            while len(self._messages) >= 2 and _estimate_tokens(self._messages) > self.max_tokens:
                self._messages = self._messages[2:]
        if self.auto_save:
            await self.save()

    async def save(self) -> Path:
        """
        Write current history to .daita/sessions/{workspace}/{session_id}.json.

        Uses an atomic write (temp file + rename) so a crash mid-save never
        leaves a corrupt file. Creates parent directories as needed.
        Returns the path written to. Raises ValueError if workspace not set.
        """
        if self.workspace is None:
            raise ValueError(
                "Cannot save: workspace is not set. Either pass workspace= at construction "
                "or call agent.run(history=history) first to derive it automatically."
            )
        path = self._resolve_path()
        tmp = path.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w") as f:
            await f.write(json.dumps(self._messages, indent=2))
        tmp.rename(path)
        return path

    @classmethod
    async def load(
        cls,
        session_id: str,
        workspace: str,
        max_turns: Optional[int] = None,
        max_tokens: Optional[int] = None,
        auto_save: bool = False,
        scope: str = "project",
        base_dir: Optional[Path] = None,
    ) -> "ConversationHistory":
        """
        Load an existing session from disk.

        Returns an empty ConversationHistory (not an error) if the file does
        not exist or is corrupt. workspace is required here because there is
        no agent context at load time.
        """
        history = cls(
            session_id=session_id,
            workspace=workspace,
            max_turns=max_turns,
            max_tokens=max_tokens,
            auto_save=auto_save,
            scope=scope,
            base_dir=base_dir,
        )
        path = history._resolve_path()
        if path.exists():
            try:
                async with aiofiles.open(path) as f:
                    history._messages = json.loads(await f.read())
            except json.JSONDecodeError:
                logger.warning("Corrupt session file at %s, starting with empty history", path)
        return history

    def clear(self) -> None:
        """Reset history in memory only. Does not touch the file on disk."""
        self._messages = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_workspace(self, agent_id: str) -> None:
        """If workspace not already set, derive it from agent_id. Called by _run_traced()."""
        if self.workspace is None:
            self.workspace = _derive_workspace(agent_id)

    def _resolve_path(self) -> Path:
        """Compute the full file path. Raises ValueError if workspace is None."""
        if self.workspace is None:
            raise ValueError(
                "Cannot resolve path: workspace is not set. Either pass workspace= at "
                "construction or call agent.run(history=history) first."
            )
        if self._base_dir is not None:
            base = self._base_dir
        elif self._scope == "global":
            base = Path.home() / ".daita" / "sessions"
        else:
            project_root = _find_project_root()
            base = (project_root or Path.cwd()) / ".daita" / "sessions"

        path = base / self.workspace / f"{self.session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
