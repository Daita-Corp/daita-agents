"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from tests.tui._support import (
    Agent,
    DaitaApp,
    Iterator,
    Path,
    contextmanager,
    workspace_for,
)


def test_external_editor_runs_only_inside_textual_suspend():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    events: list[str] = []

    @contextmanager
    def suspended() -> Iterator[None]:
        events.append("suspend")
        try:
            yield
        finally:
            events.append("resume")

    app.suspend = suspended  # type: ignore[method-assign]

    def edit(seed: str) -> str:
        events.append(f"edit:{seed}")
        return seed + " changed"

    app.controller.edit_document = edit  # type: ignore[method-assign]
    assert app._edit_document("memory") == "memory changed"
    assert events == ["suspend", "edit:memory", "resume"]


async def test_skill_and_memory_editor_flows_use_public_controller(
    tmp_path: Path,
):
    opened = await Agent.create(
        "editors", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        app._edit_document = lambda seed: (  # type: ignore[method-assign]
            "# audit\n\nUse for audit work.\n\n"
            "## Instructions\n\nCheck every recorded total.\n"
        )
        await app._create_skill("audit")
        skill = await opened.read_skill("audit")
        assert skill is not None
        assert skill.description == "Use for audit work."

        app._edit_document = lambda seed: seed + "Remember fiscal calendars.\n"  # type: ignore[method-assign]
        await app._edit_memory_target("memory")
        assert "Remember fiscal calendars." in await opened.read_memory()
    finally:
        await opened.close()
