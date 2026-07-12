from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def _phase4_sources() -> tuple[Path, ...]:
    return (
        *sorted((ROOT / "daita/db/monitor_commands").glob("*.py")),
        *sorted((ROOT / "daita/db/runtime").glob("monitor*.py")),
        *sorted((ROOT / "daita/db/runtime/monitor_actions").glob("*.py")),
        ROOT / "daita/db/runtime/extensions/memory_learning.py",
        ROOT / "daita/db/runtime/extensions/memory_update.py",
    )


def test_phase4_workflow_sources_do_not_plan_kernel_tasks_or_construct_raw_tasks():
    offenders: list[str] = []
    for path in _phase4_sources():
        source = path.read_text()
        if "kernel.plan_task" in source:
            offenders.append(f"{path.relative_to(ROOT)}: kernel.plan_task")
        if re.search(r"\bTask\(", source):
            offenders.append(f"{path.relative_to(ROOT)}: Task(")

    assert offenders == []
