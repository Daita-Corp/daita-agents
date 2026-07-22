import ast
from pathlib import Path
import sqlite3

import daita

from daita.capabilities import AccessMode
from daita.storage.sqlite import SQLiteStateStore

PACKAGE = Path(daita.__file__).parent
ROOT = PACKAGE.parents[1]


def _python_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.py"))
    )


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"missing class {class_name} in {path}")


def _class_owners(class_name: str) -> set[str]:
    owners = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.ClassDef) and node.name == class_name
            for node in ast.walk(tree)
        ):
            owners.add(path.relative_to(PACKAGE).as_posix())
    return owners


def test_final_src_layout_has_one_package_owner_and_no_replacement_alias():
    assert PACKAGE == ROOT / "src" / "daita"
    assert not (ROOT / "daita").exists()
    assert not (ROOT / "next").exists()
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'package-dir = {"" = "src"}' in packaging
    assert 'daita = "daita.cli:main"' in packaging


def test_public_surface_is_focused():
    assert set(daita.__all__) == {
        "Agent",
        "AgentConfig",
        "AgentEvent",
        "AgentEventKind",
        "AgentObserver",
        "ApprovalDecision",
        "ApprovalHandler",
        "ApprovalRequest",
        "ConversationRun",
        "LocalDirectorySource",
        "LoopExit",
        "LoopExitKind",
        "LoopLimits",
        "ModelRoute",
        "ModelRouteCandidate",
        "PostgreSQLSource",
        "RetryPolicy",
        "SQLiteSource",
        "Skill",
        "SkillSummary",
        "Transcript",
        "__version__",
        "create_llm_provider",
    }


def test_stage_seven_exports_records_without_exporting_their_owners():
    assert daita.ConversationRun.__module__ == "daita.loop.models"
    assert daita.AgentEvent.__module__ == "daita.observation"
    assert daita.AgentEventKind.__module__ == "daita.observation"
    assert daita.AgentObserver.__module__ == "daita.observation"
    assert daita.ApprovalDecision.__module__ == "daita.capabilities"
    assert daita.ApprovalRequest.__module__ == "daita.capabilities"
    assert daita.ApprovalHandler.__module__ == "daita.capabilities"
    assert daita.Skill.__module__ == "daita.skills.store"
    assert daita.SkillSummary.__module__ == "daita.skills.store"

    assert set(daita.__all__).isdisjoint(
        {
            "CapabilityRegistry",
            "MemoryStore",
            "SideEffectExecutor",
            "SkillStore",
            "_emit_safely",
        }
    )


def test_stage_seven_survivor_docs_and_examples_describe_only_the_mvp():
    root = PACKAGE.parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    examples_readme = (root / "examples" / "README.md").read_text(encoding="utf-8")
    plan = (
        root / "docs" / "MVP_MEMORY_SKILLS_GOVERNANCE_OBSERVABILITY_PLAN_2026-07-21.md"
    ).read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_examples_readme = " ".join(examples_readme.split())

    for required in (
        "at most 8 runs, 40 messages, and 24,000 UTF-8 bytes",
        "Data access remains read-only",
        "foreground",
        "in-process approve-once callback",
        "does not persist events, collect telemetry",
        "session runtime",
    ):
        assert required in normalized_readme
    assert "Implemented through Stage 7" in plan
    assert "bounded cold continuation" in normalized_examples_readme
    assert "best-effort non-persisted events" in normalized_examples_readme

    example_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "examples").glob("*.py"))
    )
    for deleted_import in (
        "daita.events",
        "daita.extensions",
        "daita.monitors",
        "daita.operations",
        "daita.sessions",
        "daita.telemetry",
    ):
        assert deleted_import not in example_text


def test_deleted_lifecycle_systems_remain_absent():
    for package_name in (
        "events",
        "extensions",
        "monitors",
        "operations",
        "telemetry",
    ):
        assert not any((PACKAGE / package_name).glob("*.py"))
    for module_name in (
        "approvals.py",
        "governance.py",
        "learning.py",
        "sessions.py",
    ):
        assert not (PACKAGE / module_name).exists()


def test_future_memory_skill_and_observation_surfaces_are_slim():
    allowed_package_files = {"__init__.py", "capabilities.py", "store.py"}
    deleted_lifecycle_files = {
        "context.py",
        "learning.py",
        "models.py",
        "protocols.py",
        "service.py",
        "worker.py",
    }
    for package_name in ("memory", "skills"):
        candidate = PACKAGE / package_name
        if not candidate.exists():
            continue
        files = {path.name for path in candidate.glob("*.py")}
        assert files <= allowed_package_files
        assert files.isdisjoint(deleted_lifecycle_files)

    observation = PACKAGE / "observation.py"
    if observation.exists():
        text = observation.read_text(encoding="utf-8").lower()
        for term in (
            "background worker",
            "dispatcher",
            "event bus",
            "event store",
            "exporter",
            "middleware",
            "observer queue",
            "subscription",
            "telemetry sink",
            "trace tree",
        ):
            assert term not in text


def test_stage_five_memory_keeps_storage_separate_from_one_fixed_write_identity():
    memory = PACKAGE / "memory"
    assert {path.name for path in memory.glob("*.py")} == {
        "__init__.py",
        "capabilities.py",
        "store.py",
    }
    store_text = (memory / "store.py").read_text(encoding="utf-8")
    capability_text = (memory / "capabilities.py").read_text(encoding="utf-8")
    assert '"MEMORY.md"' in store_text
    assert '"USER.md"' in store_text
    for term in (
        "Capability(",
        "Executor(",
        "ToolView(",
        "memory_set",
        "sqlite",
        "catalog",
        "approval",
        "policy",
        "telemetry",
    ):
        assert term not in store_text

    assert 'MEMORY_SET_TOOL_NAME = "memory_set"' in capability_text
    assert 'MEMORY_SET_CAPABILITY_ID = "memory.set"' in capability_text
    assert 'MEMORY_SET_EXECUTOR_ID = "memory.set.executor"' in capability_text
    assert 'MEMORY_SET_OUTPUT_KIND = "memory.replacement"' in capability_text
    for forbidden in ("skill_save", "skill_delete", "SQLiteStateStore"):
        assert forbidden not in capability_text

    registry_text = (PACKAGE / "capabilities.py").read_text(encoding="utf-8")
    controller_text = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    assert "daita.memory" not in registry_text
    assert "MemoryStore" not in registry_text
    assert "MemoryStore" not in controller_text

    expected = {
        "read_memory",
        "set_memory",
        "read_user_profile",
        "set_user_profile",
    }
    assert expected <= _class_methods(PACKAGE / "agent.py", "Agent")
    assert expected <= _class_methods(
        PACKAGE / "hosting" / "embedded.py", "EmbeddedAgent"
    )


def test_stage_six_skills_extend_the_slim_progressive_owner_with_two_writes():
    skills = PACKAGE / "skills"
    assert {path.name for path in skills.glob("*.py")} == {
        "__init__.py",
        "capabilities.py",
        "store.py",
    }
    store_text = (skills / "store.py").read_text(encoding="utf-8")
    capability_text = (skills / "capabilities.py").read_text(encoding="utf-8")
    assert '"skills"' in store_text
    assert '"SKILL.md"' in store_text
    assert 'SKILL_VIEW_TOOL_NAME = "skill_view"' in capability_text
    assert 'SKILL_VIEW_CAPABILITY_ID = "skill.view"' in capability_text
    assert 'SKILL_VIEW_EXECUTOR_ID = "skill.view.executor"' in capability_text
    assert 'SKILL_VIEW_OUTPUT_KIND = "skill.document"' in capability_text
    assert 'SKILL_SAVE_TOOL_NAME = "skill_save"' in capability_text
    assert 'SKILL_SAVE_CAPABILITY_ID = "skill.save"' in capability_text
    assert 'SKILL_SAVE_EXECUTOR_ID = "skill.save.executor"' in capability_text
    assert 'SKILL_SAVE_OUTPUT_KIND = "skill.saved"' in capability_text
    assert 'SKILL_DELETE_TOOL_NAME = "skill_delete"' in capability_text
    assert 'SKILL_DELETE_CAPABILITY_ID = "skill.delete"' in capability_text
    assert 'SKILL_DELETE_EXECUTOR_ID = "skill.delete.executor"' in capability_text
    assert 'SKILL_DELETE_OUTPUT_KIND = "skill.deleted"' in capability_text
    for forbidden in (
        "CapabilityRegistry",
        "CatalogResource",
        "SQLiteStateStore",
    ):
        assert forbidden not in store_text
    assert "SideEffectExecutor" not in store_text

    expected = {"list_skills", "read_skill", "save_skill", "delete_skill"}
    assert expected <= _class_methods(PACKAGE / "agent.py", "Agent")
    assert expected <= _class_methods(
        PACKAGE / "hosting" / "embedded.py", "EmbeddedAgent"
    )

    controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    context = (PACKAGE / "domains" / "data" / "context.py").read_text(encoding="utf-8")
    assert "SKILL_VIEW_CAPABILITY_ID" in controller
    assert "SKILL_SAVE_CAPABILITY_ID" in controller
    assert "SKILL_DELETE_CAPABILITY_ID" in controller
    assert "skill_index" in context
    assert "historical skill body redacted" in context


async def test_every_composed_builtin_write_uses_preflight_and_one_runtime_branch(
    tmp_path,
):
    agent = await daita.Agent.create("write-architecture", root=tmp_path)
    try:
        registry = agent._embedded._capabilities
        write_tools = set()
        for name in registry.tool_names:
            _, capability = registry.resolve_tool(name)
            if capability.access_mode is not AccessMode.WRITE:
                continue
            write_tools.add(name)
            _, executor = registry.resolve_execution(capability.id)
            assert capability.side_effecting is True
            assert callable(getattr(executor, "preflight", None))
        assert write_tools == {"memory_set", "skill_save", "skill_delete"}
    finally:
        await agent.close()


def test_observation_owners_keep_tool_events_out_of_loop_and_storage():
    storage = _python_text(PACKAGE / "storage")
    controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    loop = (PACKAGE / "loop" / "driver.py").read_text(encoding="utf-8")

    assert "AgentEvent" not in storage
    assert "AgentEventKind.TOOL_STARTED" in controller
    assert "AgentEventKind.TOOL_COMPLETED" in controller
    assert "AgentEventKind.APPROVAL_REQUESTED" in controller
    assert "AgentEventKind.APPROVAL_DECIDED" in controller
    assert "AgentEventKind.TOOL_STARTED" not in loop
    assert "AgentEventKind.TOOL_COMPLETED" not in loop
    assert "AgentEventKind.APPROVAL_REQUESTED" not in loop
    assert "AgentEventKind.APPROVAL_DECIDED" not in loop


def test_stage_five_governance_extends_existing_execution_and_composition_owners():
    contracts = (PACKAGE / "capabilities.py").read_text(encoding="utf-8")
    controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    embedded = (PACKAGE / "hosting" / "embedded.py").read_text(encoding="utf-8")
    loop = _python_text(PACKAGE / "loop").lower()
    storage = _python_text(PACKAGE / "storage").lower()

    for contract in (
        "class ApprovalDecision",
        "class ApprovalRequest",
        "class ApprovalHandler",
        "class SideEffectExecutor",
    ):
        assert contract in contracts
    assert controller.count("side_effect.preflight(execution)") == 2
    assert "async with self._mutation_lock" in controller
    assert "state_changed" in controller
    assert "_execute_definitely" in controller
    assert embedded.count("mutation_lock = asyncio.Lock()") == 1
    assert "mutation_lock=mutation_lock" in embedded
    assert "approval_handler=approval_handler" in embedded
    assert "approval" not in loop
    assert "approval" not in storage


def test_conversations_add_grouping_without_a_runtime_or_history_system():
    text = _python_text(PACKAGE)
    for term in (
        "BackgroundWorker",
        "CompressionCheckpoint",
        "ConversationManager",
        "ConversationRuntime",
        "ConversationSearch",
        "ConversationSummary",
        "ConversationWorker",
        "EventStore",
        "PendingApproval",
        "PersistedApproval",
        "ResumeRuntime",
        "ResumeState",
        "SearchIndex",
        "SessionManager",
    ):
        assert term not in text


def test_cli_remains_a_presentation_over_the_public_agent_api():
    path = PACKAGE / "cli.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    forbidden_import_roots = {
        "adapters",
        "capabilities",
        "catalog",
        "domains",
        "hosting",
        "loop",
        "storage",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_import_roots
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "sqlite3" for alias in node.names)

    for forbidden in (
        "._embedded",
        "CapabilityRegistry",
        "DataToolRuntime",
        "executor.execute(",
        "resolve_execution(",
        "SQLiteStateStore",
    ):
        assert forbidden not in text

    public_methods = {
        method
        for method in _class_methods(PACKAGE / "agent.py", "Agent")
        if not method.startswith("_")
    }
    agent_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "agent"
    }
    assert agent_calls <= public_methods


def test_cli_adds_no_parallel_state_approval_or_observation_owner():
    cli_tree = ast.parse((PACKAGE / "cli.py").read_text(encoding="utf-8"))
    cli_classes = {
        node.name for node in ast.walk(cli_tree) if isinstance(node, ast.ClassDef)
    }
    assert cli_classes.isdisjoint(
        {
            "ApprovalService",
            "ApprovalStore",
            "CommandRegistry",
            "ConversationRuntime",
            "EventDispatcher",
            "EventStore",
            "Session",
            "SessionManager",
        }
    )

    conversation_state_fields = {
        node.target.id
        for class_node in ast.walk(cli_tree)
        if isinstance(class_node, ast.ClassDef)
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and "conversation" in node.target.id
    }
    assert conversation_state_fields <= {"conversation_id"}
    assert _class_owners("ApprovalHandler") == {"capabilities.py"}
    assert _class_owners("AgentObserver") == {"observation.py"}


def test_loop_has_no_cross_cutting_lifecycle_responsibilities():
    text = _python_text(PACKAGE / "loop").lower()
    for term in (
        "approval",
        "checkpoint",
        "governance",
        "learning",
        "readiness",
        "repair",
        "resume",
        "schema_version",
        "schema-version",
    ):
        assert term not in text


def test_new_mvp_owners_have_no_version_or_compatibility_framework():
    candidates = [PACKAGE / "loop", PACKAGE / "hosting", PACKAGE / "storage"]
    candidates.extend(
        path
        for path in (PACKAGE / "memory", PACKAGE / "skills", PACKAGE / "observation.py")
        if path.exists()
    )
    text = "\n".join(
        _python_text(path) if path.is_dir() else path.read_text(encoding="utf-8")
        for path in candidates
    ).lower()
    for term in (
        "compatibility decoder",
        "compatibility migration",
        "migration framework",
        "schema_version",
        "schema-version",
    ):
        assert term not in text


async def test_sqlite_table_set_and_conversation_grouping_are_minimal(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {
            table: tuple(
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            )
            for table in tables
        }
        named_indexes = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT name, tbl_name FROM sqlite_master "
                "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
            )
        }

        assert tables == {
            "messages",
            "metadata",
            "runs",
            "snapshots",
            "sources",
            "syncs",
        }
        assert columns == {
            "messages": ("run_id", "position", "data"),
            "metadata": ("key", "data"),
            "runs": (
                "id",
                "agent_id",
                "conversation_id",
                "turn_index",
                "input",
                "result",
            ),
            "snapshots": ("agent_id", "source_id", "sync_id", "data"),
            "sources": ("agent_id", "id", "data"),
            "syncs": ("agent_id", "id", "source_id", "data"),
        }
        assert named_indexes == {"runs_conversation_turn": "runs"}
        run_indexes = {
            row[1]: bool(row[2])
            for row in connection.execute("PRAGMA index_list(runs)")
            if not str(row[1]).startswith("sqlite_autoindex")
        }
        assert run_indexes == {"runs_conversation_turn": True}
        index_columns = tuple(
            row[2]
            for row in connection.execute("PRAGMA index_info(runs_conversation_turn)")
        )
        assert index_columns == ("agent_id", "conversation_id", "turn_index")


def test_knowledge_content_cannot_redefine_data_or_execution_authority():
    catalog_text = _python_text(PACKAGE / "catalog")
    assert "daita.memory" not in catalog_text
    assert "daita.skills" not in catalog_text
    assert "..memory" not in catalog_text
    assert "..skills" not in catalog_text

    forbidden_store_terms = (
        "Approval",
        "Capability(",
        "CapabilityRegistry",
        "CatalogResource",
        "Executor",
        "Policy",
        "ToolView(",
    )
    for package_name in ("memory", "skills"):
        store = PACKAGE / package_name / "store.py"
        if not store.exists():
            continue
        text = store.read_text(encoding="utf-8")
        for term in forbidden_store_terms:
            assert term not in text


def test_skill_save_delete_cannot_mutate_registered_execution_identities():
    registry_methods = _class_methods(PACKAGE / "capabilities.py", "CapabilityRegistry")
    assert registry_methods.isdisjoint(
        {"add", "delete", "register", "remove", "save", "unregister", "update"}
    )

    skill_store = PACKAGE / "skills" / "store.py"
    if skill_store.exists():
        text = skill_store.read_text(encoding="utf-8")
        for term in ("CapabilityRegistry", "Executor", "ToolView"):
            assert term not in text


def test_registry_and_data_runtime_keep_executor_resolution_ownership():
    resolution_owners = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "resolve_execution(" in path.read_text(encoding="utf-8")
    }
    resolved_executor_callers = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "executor.execute(" in path.read_text(encoding="utf-8")
    }
    assert resolution_owners == {"capabilities.py", "domains/data/controller.py"}
    assert resolved_executor_callers == {"domains/data/controller.py"}
