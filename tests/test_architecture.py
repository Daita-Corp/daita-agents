import ast
from collections.abc import Mapping
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


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_final_src_layout_has_one_package_owner_and_no_replacement_alias():
    assert PACKAGE == ROOT / "src" / "daita"
    assert not (ROOT / "daita").exists()
    assert not (ROOT / "next").exists()
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'package-dir = {"" = "src"}' in packaging
    assert 'daita = "daita.cli:main"' in packaging


def test_model_suggestions_remain_terminal_only_presentation_metadata():
    terminal = (PACKAGE / "terminal.py").read_text(encoding="utf-8")
    assert "_MODEL_SUGGESTIONS" in terminal
    assert _class_owners("_ModelSuggestion") == {"terminal.py"}
    for owner in ("catalog", "loop", "llm", "storage"):
        text = _python_text(PACKAGE / owner)
        assert "_MODEL_SUGGESTIONS" not in text
        assert "_ModelSuggestion" not in text


def test_public_surface_is_focused():
    assert set(daita.__all__) == {
        "Agent",
        "AgentConfig",
        "AgentEvent",
        "AgentEventKind",
        "AgentObserver",
        "ArtifactDeliveryReceipt",
        "ArtifactDestination",
        "ArtifactError",
        "ArtifactPayload",
        "ArtifactRef",
        "ApprovalDecision",
        "ApprovalHandler",
        "ApprovalRequest",
        "ConversationRun",
        "CatalogSummary",
        "DocumentCandidateContent",
        "LearningCandidate",
        "LearningCandidateAction",
        "LearningCandidateError",
        "LearningCandidateRejectionReason",
        "LearningCandidateStatus",
        "LearningCandidateTarget",
        "LearningCandidateView",
        "LearningReviewResult",
        "LearningReviewStatus",
        "LocalDirectorySource",
        "LoopExit",
        "LoopExitKind",
        "LoopLimits",
        "ModelRoute",
        "ModelRouteCandidate",
        "PostgreSQLSource",
        "RetryPolicy",
        "ResourceRevisionBinding",
        "SQLiteSource",
        "SemanticAnnotation",
        "SemanticAnnotationState",
        "SemanticAnnotationView",
        "SemanticDigestMismatchError",
        "SemanticEvidence",
        "SemanticEvidenceKind",
        "SemanticFieldReference",
        "SemanticKind",
        "SemanticSubject",
        "SemanticValidationError",
        "SemanticCandidateContent",
        "Skill",
        "SkillCandidateContent",
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
    assert daita.SemanticAnnotation.__module__ == "daita.semantics"
    assert daita.SemanticAnnotationView.__module__ == "daita.semantics"
    assert daita.ArtifactRef.__module__ == "daita.artifacts.models"
    assert daita.ArtifactPayload.__module__ == "daita.artifacts.models"
    assert daita.ArtifactDeliveryReceipt.__module__ == "daita.artifacts.models"
    assert daita.ArtifactDestination.__module__ == "daita.artifacts.models"
    assert daita.ArtifactError.__module__ == "daita.artifacts.models"

    assert set(daita.__all__).isdisjoint(
        {
            "CapabilityRegistry",
            "AgentHomeArtifactStore",
            "ArtifactDraft",
            "ArtifactPolicy",
            "LocalArtifactDelivery",
            "MemoryStore",
            "SideEffectExecutor",
            "SkillStore",
            "_emit_safely",
        }
    )


def test_survivor_docs_and_examples_describe_only_the_mvp():
    root = PACKAGE.parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    examples_readme = (root / "examples" / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_readme_words = normalized_readme.replace("-", " ")
    normalized_examples_readme = " ".join(examples_readme.split())

    for required in (
        "at most 8 runs, 40 messages, and 24,000 UTF-8 bytes",
        "Data access remains read-only",
        "foreground",
        "in-process approve-once callback",
        "does not persist events, collect telemetry",
        "session runtime",
    ):
        assert required.replace("-", " ") in normalized_readme_words
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


def test_phase_two_semantics_extend_existing_storage_context_and_runtime_owners():
    semantics = (PACKAGE / "semantics.py").read_text(encoding="utf-8")
    storage = (PACKAGE / "storage" / "sqlite.py").read_text(encoding="utf-8")
    controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    context = (PACKAGE / "domains" / "data" / "context.py").read_text(encoding="utf-8")
    embedded = (PACKAGE / "hosting" / "embedded.py").read_text(encoding="utf-8")
    terminal = (PACKAGE / "terminal.py").read_text(encoding="utf-8")

    assert _class_owners("SemanticAnnotation") == {"semantics.py"}
    assert _class_owners("SemanticSubject") == {"semantics.py"}
    assert 'SEMANTIC_SAVE_TOOL_NAME = "semantic_save"' in semantics
    assert 'SEMANTIC_DELETE_TOOL_NAME = "semantic_delete"' in semantics
    assert "CREATE TABLE IF NOT EXISTS semantic_annotations" in storage
    assert "semantic_resource_facts" in controller
    assert "semantic_annotation_issue" in controller
    assert "bind_current_semantic_evidence" in controller
    assert "without_runtime_owned_semantic_evidence" in controller
    assert "render_semantic_recall" in context
    assert "semantic_declarations(identity.id, store)" in embedded
    assert "mutation_lock=mutation_lock" in embedded
    assert "/memory [list|show <id>|edit [id]|accept <id>|" in terminal
    assert "/knowledge" not in _python_text(PACKAGE)

    expected = {
        "list_semantic_annotations",
        "read_semantic_annotation",
        "save_semantic_annotation",
        "delete_semantic_annotation",
    }
    assert expected <= _class_methods(PACKAGE / "agent.py", "Agent")
    assert expected <= _class_methods(
        PACKAGE / "hosting" / "embedded.py", "EmbeddedAgent"
    )
    for forbidden in (
        "BackgroundAgentLoop",
        "KnowledgeGraph",
        "LearningRuntime",
        "MemoryProvider",
        "SemanticExecutorKernel",
        "VectorStore",
        "reviewed_document",
    ):
        assert forbidden not in _python_text(PACKAGE)


def test_phase_three_is_read_time_maintenance_and_caller_owned_evaluation_only():
    semantics = (PACKAGE / "semantics.py").read_text(encoding="utf-8")
    context = (PACKAGE / "domains" / "data" / "context.py").read_text(encoding="utf-8")
    controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    storage = (PACKAGE / "storage" / "sqlite.py").read_text(encoding="utf-8")
    evaluation = (PACKAGE / "evaluation.py").read_text(encoding="utf-8")
    candidates = (PACKAGE / "learning_candidates.py").read_text(encoding="utf-8")
    package_text = _python_text(PACKAGE)

    assert _class_owners("AgentLoop") == {"loop/driver.py"}
    assert _class_owners("DataToolRuntime") == {"domains/data/controller.py"}
    assert _class_owners("SQLiteStateStore") == {"storage/sqlite.py"}
    assert "semantic_duplicate_identity" in semantics
    assert "SEMANTIC_MAINTENANCE_MAX_NOTICES" in semantics
    assert "semantic-maintenance" in semantics
    assert "review material only" in context
    assert "_decorate_semantic_view" in controller
    assert "_semantic_management_requested" in controller
    assert "_semantic_maintenance_requested" in controller
    assert "capability.id in _SEMANTIC_CAPABILITIES" in controller
    assert "semantic_annotations" in storage
    assert "CREATE TABLE IF NOT EXISTS learning_candidates" in storage
    assert "tools=()" in candidates
    assert "AgentLoop" not in candidates
    assert "DataToolRuntime" not in candidates
    assert "data_query_sqlite" not in candidates
    assert "data_query_postgresql" not in candidates
    assert "evaluation" not in storage.lower()
    assert "telemetry" not in storage.lower()
    assert "from .storage" not in evaluation
    assert "import sqlite3" not in evaluation.lower()
    assert "raw_prompt" not in evaluation.lower()
    assert "tool_arguments" not in evaluation.lower()

    for forbidden in (
        "class BackgroundReviewer",
        "class CandidateStore",
        "class LearningRuntime",
        "class ReviewScheduler",
        "class SemanticExecutorKernel",
        "class TelemetryStore",
        "class VectorRetriever",
        "CREATE TABLE telemetry",
        "/review-learning",
        "/knowledge",
    ):
        assert forbidden not in package_text


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
        assert write_tools == {
            "artifact_save_local",
            "artifact_set_export_location",
            "memory_set",
            "semantic_delete",
            "semantic_save",
            "skill_save",
            "skill_delete",
        }
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


def test_terminal_application_remains_a_presentation_over_the_public_agent_api():
    path = PACKAGE / "terminal.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    forbidden_import_roots = {
        "adapters",
        "capabilities",
        "catalog",
        "domains",
        "hosting",
        "llm",
        "storage",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_import_roots
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] not in {"asyncpg", "keyring", "sqlite3"}
                for alias in node.names
            )

    for forbidden in (
        "._embedded",
        "AgentLoop",
        "CapabilityRegistry",
        "DataToolRuntime",
        "ModelProvider",
        "ResourceAdapter",
        "SQLiteStateStore",
        "agent.toml",
        "state.db",
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
    assert {
        "catalog_preview",
        "conversation_exists",
        "refresh_source",
    } <= public_methods
    assert text.count("agent.run(") == 1

    terminal_classes = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert terminal_classes.isdisjoint(
        {
            "CommandRegistry",
            "ConversationRuntime",
            "ReadinessService",
            "Session",
            "SessionManager",
            "Workflow",
        }
    )


def test_terminal_presentation_modules_are_the_only_lazy_prompt_toolkit_owners():
    owners = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "prompt_toolkit" in path.read_text(encoding="utf-8")
    }
    assert owners == {"terminal_selection.py", "terminal_tui.py"}

    for owner, loader in (
        ("terminal_selection.py", "_load_prompt_toolkit"),
        ("terminal_tui.py", "_load_terminal_runtime"),
    ):
        tree = ast.parse((PACKAGE / owner).read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not imported_modules.intersection(
            {
                "daita.adapters",
                "daita.capabilities",
                "daita.catalog",
                "daita.domains",
                "daita.hosting",
                "daita.loop",
                "daita.storage",
            }
        )
        top_level_imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any(
            module == "prompt_toolkit" or module.startswith("prompt_toolkit.")
            for module in top_level_imports
        )
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == loader
            and any(
                isinstance(child, ast.ImportFrom)
                and child.module
                and child.module.startswith("prompt_toolkit")
                for child in ast.walk(node)
            )
            for node in tree.body
        )


def test_rich_is_lazy_and_owned_only_by_the_focused_terminal_tui():
    owners = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if any(
            line.lstrip().startswith(("from rich", "import rich"))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert owners == {"terminal_tui.py"}

    tree = ast.parse((PACKAGE / "terminal_tui.py").read_text(encoding="utf-8"))
    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(
        module == "rich" or module.startswith("rich.") for module in top_level_imports
    )
    assert any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_load_terminal_runtime"
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module
            and child.module.startswith("rich.")
            for child in ast.walk(node)
        )
        for node in tree.body
    )


def test_schema_multi_selector_has_no_data_runtime_or_persisted_state_owner():
    selector_path = PACKAGE / "terminal_selection.py"
    selector_tree = ast.parse(selector_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(selector_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(selector_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden_fragments = (
        "adapters",
        "capabilities",
        "catalog",
        "controller",
        "executors",
        "loop",
        "postgresql",
        "storage",
    )
    assert not any(
        any(fragment in module.split(".") for fragment in forbidden_fragments)
        for module in imported_modules
    )
    public_functions = {
        node.name
        for node in selector_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"select_one", "select_many"} <= public_functions

    storage = _python_text(PACKAGE / "storage").casefold()
    for persisted_state in (
        "checked_state",
        "highlight_position",
        "onboarding_state",
        "readiness_state",
        "selected_schema",
        "selector_position",
    ):
        assert persisted_state not in storage


def test_stage_four_summary_is_catalog_owned_and_not_loop_or_storage_state():
    assert _class_owners("CatalogSummary") == {"catalog/models.py"}
    loop = _python_text(PACKAGE / "loop")
    storage = _python_text(PACKAGE / "storage")
    for field_name in (
        "active_source_count",
        "latest_successful_sync_completed_at",
        "relationship_count",
    ):
        assert field_name not in loop
    assert "readiness" not in storage.lower()


def test_catalog_schema_slice_extends_existing_catalog_and_capability_owners():
    assert _class_owners("CatalogSchemaRequest") == {"catalog/models.py"}
    service_methods = _class_methods(
        PACKAGE / "catalog" / "service.py",
        "CatalogService",
    )
    assert "schema_slice" in service_methods
    capabilities = (PACKAGE / "catalog" / "capabilities.py").read_text(encoding="utf-8")
    embedded = (PACKAGE / "hosting" / "embedded.py").read_text(encoding="utf-8")
    loop = _python_text(PACKAGE / "loop")
    assert 'name="catalog_schema"' in capabilities
    assert 'CATALOG_SCHEMA_EVIDENCE_KIND = "catalog.schema_slice"' in capabilities
    assert "catalog_declarations(identity.id, catalog_service)" in embedded
    assert "catalog_schema" not in loop
    for prohibited in (
        "CatalogSchemaCache",
        "CatalogSearchService",
        "SchemaGraph",
        "VectorStore",
    ):
        assert prohibited not in _python_text(PACKAGE)


def test_catalog_snapshot_reuse_is_private_derived_storage_state():
    assert _class_owners("CatalogSnapshotRef") == {"catalog/models.py"}
    assert "CatalogSnapshotRef" not in daita.__all__
    protocol_methods = _class_methods(
        PACKAGE / "catalog" / "protocols.py",
        "CatalogStore",
    )
    assert {
        "list_current_snapshot_refs",
        "load_current_snapshot",
    } <= protocol_methods
    storage = (PACKAGE / "storage" / "sqlite.py").read_text(encoding="utf-8")
    assert "_decoded_catalog_snapshots" in storage
    assert "CREATE TABLE IF NOT EXISTS decoded_catalog_snapshots" not in storage
    for prohibited in (
        "CatalogSchemaCache",
        "CatalogSearchService",
        "SchemaGraph",
        "VectorStore",
    ):
        assert prohibited not in _python_text(PACKAGE)


def test_catalog_indexed_retrieval_is_private_and_catalog_owned():
    protocol_methods = _class_methods(
        PACKAGE / "catalog" / "protocols.py",
        "CatalogStore",
    )
    storage_methods = _class_methods(
        PACKAGE / "storage" / "sqlite.py",
        "SQLiteStateStore",
    )
    service = (PACKAGE / "catalog" / "service.py").read_text(encoding="utf-8")
    storage = (PACKAGE / "storage" / "sqlite.py").read_text(encoding="utf-8")
    loop = _python_text(PACKAGE / "loop")

    assert "search" not in protocol_methods
    assert "search" not in storage_methods
    assert "_SourceCatalogIndex" in service
    assert "_source_indexes" in service
    assert "_compile_source_index" in service
    assert "_catalog_search_reason" not in storage
    assert "_SourceCatalogIndex" not in storage
    assert "CatalogSearchHit" not in storage
    assert "_SourceCatalogIndex" not in loop


def test_catalog_bounded_traversal_is_catalog_owned_and_not_storage_owned():
    protocol_methods = _class_methods(
        PACKAGE / "catalog" / "protocols.py",
        "CatalogStore",
    )
    storage_methods = _class_methods(
        PACKAGE / "storage" / "sqlite.py",
        "SQLiteStateStore",
    )
    service = (PACKAGE / "catalog" / "service.py").read_text(encoding="utf-8")
    storage = (PACKAGE / "storage" / "sqlite.py").read_text(encoding="utf-8")
    loop = _python_text(PACKAGE / "loop")

    assert "traverse" not in protocol_methods
    assert "traverse" not in storage_methods
    assert "load_relationships" not in protocol_methods
    assert "load_relationships" not in storage_methods
    assert "_traverse_indexes" in service
    assert "distance_by_resource" in service
    assert "parents_by_resource" in service
    assert "deque(admitted_sources)" in service
    assert "CatalogPath" not in storage
    assert "CatalogPathStep" not in storage
    assert "distance_by_resource" not in storage
    assert "CatalogTraversalRequest" not in storage
    assert "CatalogTraversalRequest" not in loop


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


def test_pricing_semantics_have_one_provider_neutral_owner():
    assert _class_owners("CostEstimate") == {"llm/pricing.py"}
    assert _class_owners("CostComponent") == {"llm/pricing.py"}
    models = (PACKAGE / "llm" / "models.py").read_text(encoding="utf-8")
    loop = _python_text(PACKAGE / "loop").lower()
    pricing = (PACKAGE / "llm" / "pricing.py").read_text(encoding="utf-8").lower()
    assert "estimated_cost_usd" not in models
    assert "cost_per_million" not in models
    for provider in ("openai", "anthropic", "gemini", "grok", "ollama"):
        assert provider not in loop
        assert provider not in pricing


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
            "learning_candidates",
            "messages",
            "metadata",
            "runs",
            "semantic_annotations",
            "snapshots",
            "sources",
            "syncs",
        }
        assert columns == {
            "learning_candidates": ("agent_id", "id", "data"),
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
            "semantic_annotations": ("agent_id", "id", "data"),
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


def test_artifacts_have_one_concrete_owner_and_no_storage_renderer_or_policy_registry():
    assert _class_owners("AgentHomeArtifactStore") == {"artifacts/store.py"}
    assert _class_owners("LocalArtifactDelivery") == {"artifacts/delivery.py"}
    assert _class_owners("ArtifactPolicy") == {"capabilities.py"}
    assert _class_owners("ArtifactDraft") == {"artifacts/models.py"}
    artifact_text = _python_text(PACKAGE / "artifacts")
    for prohibited in (
        "ArtifactStoreRegistry",
        "ArtifactRendererRegistry",
        "ArtifactPolicyRegistry",
        "ArtifactProvider",
    ):
        assert prohibited not in artifact_text


def test_exact_csv_extends_existing_adapter_capability_and_renderer_owners():
    assert _class_owners("ExactCsvRenderer") == {"artifacts/renderers.py"}
    for adapter, class_name in (
        ("sqlite_query.py", "SQLiteQueryBackend"),
        ("postgresql_query.py", "PostgreSQLQueryBackend"),
    ):
        methods = _class_methods(PACKAGE / "adapters" / adapter, class_name)
        assert "execute_exact_csv" in methods
        tree = ast.parse((PACKAGE / "adapters" / adapter).read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "execute_exact_csv"
        )
        calls = {
            node.func.id
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "project_result_rows" not in calls
        assert "_json_value" not in calls
        assert "_unique_columns" not in calls

    renderers = (PACKAGE / "artifacts" / "renderers.py").read_text(encoding="utf-8")
    exports = (PACKAGE / "domains" / "data" / "export_capabilities.py").read_text(
        encoding="utf-8"
    )
    assert "BoundedResultProjection" not in renderers
    assert "BoundedResultProjection" not in exports
    assert "ExactCsvRenderer" not in _python_text(PACKAGE / "loop")
    assert "ArtifactRendererRegistry" not in renderers


def test_exact_csv_tool_arguments_contain_query_selection_but_never_rows_or_bytes():
    from daita.domains.data.export_capabilities import (
        POSTGRESQL_CSV_EXPORT_CAPABILITY_ID,
        SQLITE_CSV_EXPORT_CAPABILITY_ID,
        artifact_extension_declarations,
    )

    declarations = artifact_extension_declarations()
    for capability in declarations.capabilities:
        if capability.id not in {
            SQLITE_CSV_EXPORT_CAPABILITY_ID,
            POSTGRESQL_CSV_EXPORT_CAPABILITY_ID,
        }:
            continue
        properties = capability.input_schema["properties"]
        assert isinstance(properties, Mapping)
        assert set(properties) == {
            "source_id",
            "sql",
            "parameters",
            "format",
            "filename",
        }
        assert set(properties).isdisjoint(
            {"rows", "content", "bytes", "provenance", "sensitivity", "path"}
        )


def test_agent_loop_carries_artifact_records_but_never_imports_renderers_delivery_or_filesystem_paths():
    path = PACKAGE / "loop" / "driver.py"
    imports = _imports(path)
    assert "pathlib" not in imports
    assert "os" not in imports
    text = path.read_text(encoding="utf-8")
    assert "artifacts.models" in text
    assert "artifacts.delivery" not in text
    assert "artifacts.renderers" not in text
    assert "AgentHomeArtifactStore" not in text
    assert "LocalArtifactDelivery" not in text


def test_artifact_delivery_uses_no_bash_shell_subprocess_or_unrestricted_file_tool():
    path = PACKAGE / "artifacts" / "delivery.py"
    text = path.read_text(encoding="utf-8")
    imports = _imports(path)
    assert "subprocess" not in imports
    assert "bash" not in text.casefold()
    assert "shell=True" not in text
    assert "os.system" not in text
    assert "general_filesystem" not in text
    assert "data.file.write" not in text


def test_artifact_payloads_and_destination_grants_never_enter_sqlite_messages_or_model_requests():
    sqlite_text = (PACKAGE / "storage" / "sqlite.py").read_text(encoding="utf-8")
    context_text = (PACKAGE / "domains" / "data" / "context.py").read_text(
        encoding="utf-8"
    )
    assert "ArtifactPayload" not in sqlite_text
    assert "ArtifactDraft" not in sqlite_text
    assert "_DestinationGrant" not in sqlite_text
    assert "ArtifactPayload" not in context_text
    assert "_DestinationGrant" not in context_text
    assert "grant_digest" not in context_text
    assert "saved_path" not in context_text


def test_local_file_read_path_no_longer_imports_or_constructs_artifact_bytes():
    for relative in (
        "adapters/local_files.py",
        "domains/data/file_capabilities.py",
    ):
        text = (PACKAGE / relative).read_text(encoding="utf-8")
        assert "ToolArtifact" not in text
        assert "ArtifactDraft" not in text
        assert "artifact=" not in text
        assert "artifacts." not in text


def test_xlsx_dependencies_are_absent_before_phase_three_and_default_integrations_remain_lazy():
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8").casefold()
    for dependency in ("openpyxl", "xlsxwriter", "pandas"):
        assert dependency not in packaging
    artifact_text = _python_text(PACKAGE / "artifacts")
    assert "openpyxl" not in artifact_text
    assert "xlsxwriter" not in artifact_text
    assert ".xlsx" not in artifact_text
