"""
File-based storage for local memory backend.

Manages MEMORY.md and daily log files.
"""

import aiofiles
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class FileStorage:
    """File-based memory storage using local filesystem."""

    def __init__(self, workspace_dir: Path, agent_id: Optional[str] = None):
        """
        Initialize file storage.

        Args:
            workspace_dir: Base directory for workspace memory files
            agent_id: Optional agent identifier for attribution in shared workspaces
        """
        self.workspace_dir = workspace_dir
        self.agent_id = agent_id
        self.logs_dir = workspace_dir / "logs"
        self.memory_file = workspace_dir / "MEMORY.md"

        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def append_to_daily_log(
        self, content: str, category: Optional[str] = None
    ) -> str:
        """
        Append to today's daily log file.

        Args:
            content: Content to append
            category: Optional category tag

        Returns:
            Path to the log file
        """
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.logs_dir / f"{today}.md"

        # Format entry with agent attribution
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n## {timestamp}"

        # Add agent attribution for shared workspaces
        if self.agent_id:
            entry += f" [{self.agent_id}]"

        # Add category if provided
        if category:
            entry += f" [{category}]"

        entry += f"\n\n{content}\n"

        # Append to file
        async with aiofiles.open(log_file, mode="a", encoding="utf-8") as f:
            await f.write(entry)

        return str(log_file)

    async def append_to_long_term(
        self, content: str, section: Optional[str] = None
    ) -> str:
        """
        Append to long-term MEMORY.md file.

        Args:
            content: Content to append (may include metadata)
            section: Optional section header

        Returns:
            Path to the memory file
        """
        # Initialize memory file if it doesn't exist
        if not self.memory_file.exists():
            async with aiofiles.open(self.memory_file, mode="w", encoding="utf-8") as f:
                await f.write("# Long-Term Memory\n\n")

        # Format entry with agent attribution
        entry = ""
        if section:
            entry += f"\n## {section}\n\n"

        # Add agent attribution for shared workspaces (inline)
        if self.agent_id:
            entry += f"*[{self.agent_id}]* "

        entry += f"{content}\n"

        # Append to file
        async with aiofiles.open(self.memory_file, mode="a", encoding="utf-8") as f:
            await f.write(entry)

        return str(self.memory_file)

    async def read_file(self, file_path: str) -> str:
        """
        Read a complete memory file.

        Args:
            file_path: Path to the file to read

        Returns:
            File contents
        """
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            return await f.read()

    async def list_files(self) -> List[Dict[str, Any]]:
        """
        List all memory files.

        Returns:
            List of file metadata dicts
        """
        files = []

        # Add MEMORY.md if exists
        if self.memory_file.exists():
            stat = self.memory_file.stat()
            files.append(
                {
                    "path": str(self.memory_file),
                    "type": "long_term",
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        # Add all daily logs
        for log_file in sorted(self.logs_dir.glob("*.md")):
            stat = log_file.stat()
            files.append(
                {
                    "path": str(log_file),
                    "type": "daily_log",
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        return files
