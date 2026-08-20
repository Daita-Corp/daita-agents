import ast
import inspect
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import daita
from daita.capabilities import AccessMode
from daita.storage.sqlite import SQLiteStateStore

PACKAGE = Path(daita.__file__).parent
ROOT = PACKAGE.parents[1]


def test_public_agent_facade_cannot_replace_composed_context_or_tool_runtime():
    for method in (daita.Agent.create, daita.Agent.open):
        parameters = inspect.signature(method).parameters
        assert "context_builder" not in parameters
        assert "tools" not in parameters


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


def test_stage_m1_has_one_common_runtime_and_no_legacy_compatibility_surface():
    prohibited = (
        "Data" + "ToolRuntime",
        "_MVP_" + "CAPABILITIES",
        "_projected_" + "tool_names",
        "Tool" + "Applicability",
        "Extension" + "Declarations",
        "extension_" + "declarations",
        "Plugin" + "Error",
        "plugin_" + "id",
        "_data_" + "tool_runtime",
        "Catalog" + "DataReader",
    )
    production = _python_text(PACKAGE)
    for symbol in prohibited:
        assert symbol not in production

    runtime_path = PACKAGE / "capability_runtime.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime)
    assert _class_owners("CapabilityRuntime") == {"capability_runtime.py"}
    assert not {
        "adapters",
        "catalog",
        "domains.data",
        "memory",
        "semantics",
        "skills",
    } & _imports(runtime_path)
    assert not {
        node.id
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Name) and node.id.endswith("_CAPABILITY_ID")
    }
    for prefix in ("artifact.", "catalog.", "data.", "memory.", "semantic.", "skill."):
        assert prefix not in runtime


def test_stage_m1_keeps_loop_context_and_composition_owners_exact():
    embedded = (PACKAGE / "hosting" / "embedded.py").read_text(encoding="utf-8")
    loop = (PACKAGE / "loop" / "driver.py").read_text(encoding="utf-8")
    assert _class_owners("DataContextBuilder") == {"domains/data/context.py"}
    assert _class_owners("ToolRuntime") == {"loop/driver.py"}
    assert "tools: ToolRuntime" in loop
    assert "_capability_runtime" in embedded
    assert "CapabilityRuntime(" in embedded
    assert "capability_runtime" not in loop


def test_stage_m2_is_server_neutral_lazy_and_uses_existing_runtime_owners():
    runtime = (PACKAGE / "capability_runtime.py").read_text(encoding="utf-8")
    adapter_path = PACKAGE / "adapters" / "mcp.py"
    adapter = adapter_path.read_text(encoding="utf-8")
    domain = (PACKAGE / "domains" / "mcp.py").read_text(encoding="utf-8")
    embedded = (PACKAGE / "hosting" / "embedded.py").read_text(encoding="utf-8")
    production = _python_text(PACKAGE)

    assert "MCP" not in runtime
    assert "mcp" not in _imports(PACKAGE / "capability_runtime.py")
    adapter_tree = ast.parse(adapter)
    top_level_imports = {
        alias.name.split(".")[0]
        for node in adapter_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "httpx" not in top_level_imports
    assert _class_owners("StreamableHTTPMCPClient") == {"adapters/mcp.py"}
    assert _class_owners("MCPCapabilityDomain") == {"domains/mcp.py"}
    assert "CapabilityRuntime(" not in adapter
    assert "CapabilityRuntime(" not in domain
    assert "activate_mcp_domain" in embedded
    assert "mcp_domain" in embedded
    assert "mcp_server_bindings" in (
        PACKAGE / "storage" / "sqlite_schema.py"
    ).read_text(encoding="utf-8")

    for fixture_only in (
        "alpha.fixture.test",
        "beta.fixture.test",
        "fixture-alpha",
        "fixture-beta",
    ):
        assert fixture_only not in production
    for prohibited in (
        "SupportedMCPServer",
        "MCPRuntime",
        "MCPRegistry",
        "MCPContextBuilder",
        "singleton_mcp",
        "built_in_endpoint",
    ):
        assert prohibited not in production

    public_operations = {
        "inspect_mcp_server",
        "attach_mcp_server",
        "list_mcp_servers",
        "refresh_mcp_server",
        "revoke_mcp_server",
    }
    assert public_operations <= _class_methods(PACKAGE / "agent.py", "Agent")
    assert public_operations <= _class_methods(
        PACKAGE / "hosting" / "embedded.py", "EmbeddedAgent"
    )


async def test_stage_m1_registry_assigns_every_native_tool_to_one_static_owner(
    tmp_path,
):
    agent = await daita.Agent.create("stage-m1-owners", root=tmp_path)
    try:
        registry = agent._embedded._capabilities
        runtime = agent._embedded._capability_runtime
        expected_owners = {"artifacts", "data", "memory", "semantics", "skills"}
        assert registry.domain_owner_ids == expected_owners
        assert set(runtime._domains) == expected_owners

        resolved = {}
        for name in registry.tool_names:
            view, capability, owner_id = registry.resolve_tool_owner(name)
            assert view.capability_id == capability.id
            assert registry.resolve_domain_owner(capability.id) == owner_id
            assert capability in runtime._domains[owner_id].declarations.capabilities
            resolved[name] = owner_id

        assert resolved["catalog_search"] == "data"
        assert resolved["data_query_sqlite"] == "data"
        assert resolved["data_query_postgresql"] == "data"
        assert resolved["data_read_file"] == "data"
        assert resolved["data_update_postgresql"] == "data"
        assert resolved["memory_set"] == "memory"
        assert resolved["skill_view"] == "skills"
        assert resolved["semantic_list"] == "semantics"
        assert resolved["artifact_create_document"] == "artifacts"
    finally:
        await agent.close()


def test_final_src_layout_has_one_package_owner_and_no_replacement_alias():
    assert PACKAGE == ROOT / "src" / "daita"
    assert not (ROOT / "daita").exists()
    assert not (ROOT / "next").exists()
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'package-dir = {"" = "src"}' in packaging
    assert 'daita = "daita.cli:main"' in packaging


def test_model_suggestions_remain_terminal_only_presentation_metadata():
    models = (PACKAGE / "tui" / "models.py").read_text(encoding="utf-8")
    assert "MODEL_SUGGESTIONS" in models
    assert _class_owners("ModelSuggestion") == {"tui/models.py"}
    for owner in ("catalog", "loop", "llm", "storage"):
        text = _python_text(PACKAGE / owner)
        assert "MODEL_SUGGESTIONS" not in text
        assert "ModelSuggestion" not in text


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
        "MCPAdmissionError",
        "MCPAuthentication",
        "MCPAuthenticationMode",
        "MCPBindingState",
        "MCPBindingStatus",
        "MCPError",
        "MCPInspectedTool",
        "MCPServerBinding",
        "MCPServerInspection",
        "MCPToolBinding",
        "MCPToolSelection",
        "ModelRoute",
        "ModelRouteCandidate",
        "PostgreSQLSource",
        "PostgreSQLUpdateReadiness",
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
        "Data access is read first",
        "explicitly scoped PostgreSQL update",
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

    skill_capabilities = (PACKAGE / "skills" / "capabilities.py").read_text(
        encoding="utf-8"
    )
    context = (PACKAGE / "domains" / "data" / "context.py").read_text(encoding="utf-8")
    assert "class SkillCapabilityDomain" in skill_capabilities
    assert "SKILL_VIEW_CAPABILITY_ID" in skill_capabilities
    assert "SKILL_SAVE_CAPABILITY_ID" in skill_capabilities
    assert "SKILL_DELETE_CAPABILITY_ID" in skill_capabilities
    assert "skill_index" in context
    assert "historical skill body redacted" in context


def test_phase_two_semantics_extend_existing_storage_context_and_runtime_owners():
    semantics = (PACKAGE / "semantics.py").read_text(encoding="utf-8")
    schema = (PACKAGE / "storage" / "sqlite_schema.py").read_text(encoding="utf-8")
    context = (PACKAGE / "domains" / "data" / "context.py").read_text(encoding="utf-8")
    embedded = (PACKAGE / "hosting" / "embedded.py").read_text(encoding="utf-8")
    terminal = (PACKAGE / "terminal.py").read_text(encoding="utf-8")

    assert _class_owners("SemanticAnnotation") == {"semantics.py"}
    assert _class_owners("SemanticSubject") == {"semantics.py"}
    assert 'SEMANTIC_SAVE_TOOL_NAME = "semantic_save"' in semantics
    assert 'SEMANTIC_DELETE_TOOL_NAME = "semantic_delete"' in semantics
    assert "CREATE TABLE semantic_annotations" in schema
    assert "class SemanticCapabilityDomain" in semantics
    assert "semantic_resource_facts" in semantics
    assert "_annotation_issue" in semantics
    assert "_bind_current_evidence" in semantics
    assert "render_semantic_recall" in context
    assert "semantic_declarations(identity.id, store)" in embedded
    assert "mutation_lock=mutation_lock" in embedded
    controller = (PACKAGE / "tui" / "controller.py").read_text(encoding="utf-8")
    assert "/memory [list|show <id>|edit [id]|accept <id>|" in controller
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
    runtime = (PACKAGE / "capability_runtime.py").read_text(encoding="utf-8")
    learning = (PACKAGE / "domains" / "learning.py").read_text(encoding="utf-8")
    storage = (PACKAGE / "storage" / "sqlite.py").read_text(encoding="utf-8")
    schema = (PACKAGE / "storage" / "sqlite_schema.py").read_text(encoding="utf-8")
    evaluation = (PACKAGE / "evaluation.py").read_text(encoding="utf-8")
    candidates = (PACKAGE / "learning_candidates.py").read_text(encoding="utf-8")
    package_text = _python_text(PACKAGE)

    assert _class_owners("AgentLoop") == {"loop/driver.py"}
    assert _class_owners("CapabilityRuntime") == {"capability_runtime.py"}
    assert _class_owners("SQLiteStateStore") == {"storage/sqlite.py"}
    assert "semantic_duplicate_identity" in semantics
    assert "SEMANTIC_MAINTENANCE_MAX_NOTICES" in semantics
    assert "semantic-maintenance" in semantics
    assert "review material only" in context
    assert "_decorate_view" in semantics
    assert "select_explicit_learning_run" in semantics
    assert "_maintenance_requested" in semantics
    assert "class LearningCandidateGuard" in learning
    for capability_id in (
        "semantics.list",
        "semantics.view",
        "semantics.save",
        "semantics.delete",
    ):
        assert capability_id not in runtime
    assert "semantic_annotations" in storage
    assert "CREATE TABLE learning_candidates" in schema
    assert "tools=()" in candidates
    assert "AgentLoop" not in candidates
    assert "CapabilityRuntime" not in candidates
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
            "data_update_postgresql",
            "memory_set",
            "semantic_delete",
            "semantic_save",
            "skill_save",
            "skill_delete",
        }
    finally:
        await agent.close()


async def test_database_write_phase_three_registers_only_the_postgresql_update_slice(
    tmp_path,
):
    agent = await daita.Agent.create("database-write-phase-two", root=tmp_path)
    try:
        registry = agent._embedded._capabilities
        capability_ids = {
            registry.resolve_tool(name)[1].id for name in registry.tool_names
        }
        preview_tool = "data_preview_postgresql_update"
        preview_capability = "data.postgresql.update_impact"
        update_tool = "data_update_postgresql"
        update_capability = "data.postgresql.update"
        forbidden_tools = {
            "data_preview_sqlite_update",
            "data_update_sqlite",
        }
        forbidden_capabilities = {
            "data.sqlite.update_impact",
            "data.sqlite.update",
        }

        assert preview_tool in registry.tool_names
        preview = registry.resolve_tool(preview_tool)[1]
        assert preview.id == preview_capability
        assert preview.access_mode is AccessMode.READ
        assert preview.side_effecting is False
        assert update_tool in registry.tool_names
        update = registry.resolve_tool(update_tool)[1]
        assert update.id == update_capability
        assert update.access_mode is AccessMode.WRITE
        assert update.side_effecting is True
        _, update_executor = registry.resolve_execution(update.id)
        assert callable(getattr(update_executor, "preflight", None))
        assert forbidden_tools.isdisjoint(registry.tool_names)
        assert forbidden_capabilities.isdisjoint(capability_ids)

        controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
            encoding="utf-8"
        )
        runtime = (PACKAGE / "capability_runtime.py").read_text(encoding="utf-8")
        package_text = _python_text(PACKAGE)
        for dormant_name in forbidden_tools | forbidden_capabilities:
            assert f'"{dormant_name}"' not in package_text
            assert f"'{dormant_name}'" not in package_text
        assert "class PostgreSQLUpdateExecutor" in package_text
        write_backend = (PACKAGE / "adapters" / "postgresql_write.py").read_text(
            encoding="utf-8"
        )
        assert "start_database_write_receipt" in write_backend
        assert "finish_database_write_receipt" in write_backend
        assert "database_write_receipts" not in write_backend
        assert "SideEffectExecutor" not in write_backend
        assert "approval_handler" not in write_backend
        assert "_execute_side_effect" in runtime
        assert "ApprovalRequest" in runtime
        capabilities_owner = (
            PACKAGE / "domains" / "data" / "capabilities.py"
        ).read_text(encoding="utf-8")
        assert ".execute_update(" in capabilities_owner
        assert ".execute_update(" not in controller
        assert ".execute_update(" not in (PACKAGE / "loop" / "driver.py").read_text(
            encoding="utf-8"
        )
        embedded = (PACKAGE / "hosting" / "embedded.py").read_text(encoding="utf-8")
        assert embedded.count("mutation_lock = asyncio.Lock()") == 1
        assert "pending_database_write" not in package_text
        assert "database_write_events" not in package_text
        assert "DbRuntime" not in package_text
        assert "RuntimeKernel" not in package_text
        for method in (
            "inspect_source_permissions",
            "preview_source_permissions",
            "apply_source_permissions",
        ):
            assert method in _class_methods(PACKAGE / "agent.py", "Agent")
            assert method in _class_methods(
                PACKAGE / "hosting" / "embedded.py", "EmbeddedAgent"
            )
            assert method not in controller
            assert method not in (
                PACKAGE / "domains" / "data" / "context.py"
            ).read_text(encoding="utf-8")
    finally:
        await agent.close()


def test_database_write_phase_four_control_plane_keeps_current_owners():
    agent_methods = _class_methods(PACKAGE / "agent.py", "Agent")
    embedded_methods = _class_methods(
        PACKAGE / "hosting" / "embedded.py",
        "EmbeddedAgent",
    )
    backend_methods = _class_methods(
        PACKAGE / "adapters" / "postgresql_write.py",
        "PostgreSQLUpdatePreviewBackend",
    )
    controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    context = (PACKAGE / "domains" / "data" / "context.py").read_text(encoding="utf-8")
    cli = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    terminal = (PACKAGE / "terminal.py").read_text(encoding="utf-8")
    tui_commands = (PACKAGE / "tui" / "commands.py").read_text(encoding="utf-8")
    tui_controller = (PACKAGE / "tui" / "controller.py").read_text(encoding="utf-8")

    assert "postgresql_update_readiness" in agent_methods
    assert "postgresql_update_readiness" in embedded_methods
    assert "postgresql_update_readiness" in backend_methods
    assert _class_owners("PostgreSQLUpdateReadiness") == {
        "adapters/postgresql_write.py"
    }
    assert "postgresql_update_readiness" not in controller
    assert "postgresql_update_readiness" not in context
    assert ".postgresql_update_readiness(" in cli
    assert ".postgresql_update_readiness(" not in terminal
    assert "inspect_source_permissions" in tui_controller
    assert "preview_source_permissions" in tui_controller
    assert "apply_source_permissions" in tui_controller
    assert '"/source permissions"' in tui_commands
    for obsolete_terminal_command in (
        "/source write inspect",
        "/source write enable",
        "/source write disable",
        "/source write readiness",
    ):
        assert obsolete_terminal_command not in terminal
        assert obsolete_terminal_command not in tui_commands
    production = _python_text(PACKAGE)
    for administration in (
        "CREATE ROLE daita_writer",
        "GRANT CONNECT ON DATABASE",
        "administrator_password",
    ):
        assert administration not in production
    for later_phase in (
        "data_insert_postgresql",
        "data_delete_postgresql",
        "execute_postgresql_sql",
        "reconcile_database_write",
    ):
        assert later_phase not in production


def test_phase_c_removes_legacy_permission_runtime_but_retains_migration_evidence():
    repository_text = (
        "\n".join(
            path.read_text(encoding="utf-8")
            for root in (PACKAGE, ROOT / "tests", ROOT / "docs", ROOT / "examples")
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".md"}
        )
        + (ROOT / "README.md").read_text(encoding="utf-8")
        + (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    )
    for removed in (
        "/source " + "config",
        "set_source_" + "write_access",
        "_configure_postgresql_" + "source",
        "source-write-" + "access",
        "clear_postgresql_" + "update_scopes",
        "required_configuration_" + "flags",
    ):
        assert removed not in repository_text

    historical_terms = ("postgresql_write_" + "admissions", "write_" + "access")
    matched = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if any(term in path.read_text(encoding="utf-8") for term in historical_terms)
    }
    assert matched == {
        "storage/sqlite_codecs/sources.py",
        "storage/sqlite_migrations/postgresql_write_admission.py",
        "storage/sqlite_migrations/scoped_source_permissions.py",
        "storage/sqlite_schema.py",
    }


def test_artifact_continuity_replaces_prompt_routing_and_history_refs_once():
    controller = (PACKAGE / "domains" / "data" / "controller.py").read_text(
        encoding="utf-8"
    )
    context = (PACKAGE / "domains" / "data" / "context.py").read_text(encoding="utf-8")
    exports = (PACKAGE / "domains" / "data" / "export_capabilities.py").read_text(
        encoding="utf-8"
    )
    for obsolete in (
        "_explicit_artifact_request",
        "_explicit_default_location_request",
        "_ARTIFACT_ACTION_WORDS",
        "_ARTIFACT_OBJECT_WORDS",
        "_intent_clauses",
    ):
        assert obsolete not in controller
    for obsolete_history_owner in (
        "ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND",
        "DOCUMENT_CREATE_OUTPUT_KIND",
        "TABULAR_EXPORT_OUTPUT_KIND",
        "LOCAL_FILE_COPY_OUTPUT_KIND",
    ):
        assert obsolete_history_owner not in context
    assert 'ARTIFACT_LIST_TOOL_NAME = "artifact_list"' in exports
    assert 'ARTIFACT_READ_TOOL_NAME = "artifact_read"' in exports
    assert 'ARTIFACT_CONVERT_TOOL_NAME = "artifact_convert"' in exports
    assert "artifact_list" not in (PACKAGE / "agent.py").read_text(encoding="utf-8")
    assert "artifact_list" not in (PACKAGE / "hosting" / "embedded.py").read_text(
        encoding="utf-8"
    )
    assert "artifact_list" not in (PACKAGE / "cli.py").read_text(encoding="utf-8")


def test_observation_owners_keep_tool_events_out_of_loop_and_storage():
    storage = _python_text(PACKAGE / "storage")
    runtime = (PACKAGE / "capability_runtime.py").read_text(encoding="utf-8")
    loop = (PACKAGE / "loop" / "driver.py").read_text(encoding="utf-8")

    assert "AgentEvent" not in storage
    assert "AgentEventKind.TOOL_STARTED" in runtime
    assert "AgentEventKind.TOOL_COMPLETED" in runtime
    assert "AgentEventKind.APPROVAL_REQUESTED" in runtime
    assert "AgentEventKind.APPROVAL_DECIDED" in runtime
    assert "AgentEventKind.TOOL_STARTED" not in loop
    assert "AgentEventKind.TOOL_COMPLETED" not in loop
    assert "AgentEventKind.APPROVAL_REQUESTED" not in loop
    assert "AgentEventKind.APPROVAL_DECIDED" not in loop


def test_stage_five_governance_extends_existing_execution_and_composition_owners():
    contracts = (PACKAGE / "capabilities.py").read_text(encoding="utf-8")
    runtime = (PACKAGE / "capability_runtime.py").read_text(encoding="utf-8")
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
    assert runtime.count("side_effect.preflight(execution)") == 2
    assert "async with self._mutation_lock" in runtime
    assert "state_changed" in runtime
    assert "_execute_definitely" in runtime
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
        "CapabilityRuntime",
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
    tui_root = PACKAGE / "tui"
    text = _python_text(tui_root)
    tree = ast.parse(text)

    forbidden_import_roots = {
        "adapters",
        "catalog",
        "domains",
        "hosting",
        "loop",
        "storage",
    }
    for path in tui_root.rglob("*.py"):
        module_tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if node.module.startswith("daita."):
                    root = node.module.split(".")[1]
                assert root not in forbidden_import_roots
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in {"asyncpg", "keyring", "sqlite3"}
                    for alias in node.names
                )

    for forbidden in (
        "._embedded",
        "AgentLoop",
        "CapabilityRegistry",
        "CapabilityRuntime",
        "ResourceAdapter",
        "SQLiteStateStore",
        "agent.toml",
        "state.db",
        "CommandRegistry",
        "ConversationRuntime",
        "ReadinessService",
        "SessionManager",
        "Workflow",
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
    assert not (PACKAGE / "terminal_tui.py").exists()
    assert not (PACKAGE / "terminal_selection.py").exists()
    assert not (PACKAGE / "terminal_transcript.py").exists()


def test_textual_and_rich_stay_behind_the_interactive_entry_boundary():
    owners = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "prompt_toolkit" in path.read_text(encoding="utf-8")
    }
    assert owners == set()

    textual_owners = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if any(
            line.lstrip().startswith(("from textual", "import textual"))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert textual_owners
    assert all(
        owner.startswith("tui/") or owner == "terminal.py" for owner in textual_owners
    )
    assert "cli.py" not in textual_owners
    terminal_tree = ast.parse((PACKAGE / "terminal.py").read_text(encoding="utf-8"))
    top_level = {
        node.module
        for node in terminal_tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in terminal_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "textual" not in top_level
    assert not any(
        module == "textual" or (module or "").startswith("textual.")
        for module in top_level
    )
    tree = ast.parse((PACKAGE / "terminal.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_load_textual_app"
        for node in tree.body
    )


def test_textual_presentation_has_one_owner_per_concern():
    assert _class_owners("ClipboardResult") == {"tui/clipboard.py"}
    assert _class_owners("ToolCardState") == {"tui/models.py"}
    assert _class_owners("DaitaApp") == {"tui/app.py"}
    assert _class_owners("PresentationController") == {"tui/controller.py"}
    assert _class_owners("ApprovalPanel") == {"tui/widgets/approval.py"}
    assert _class_owners("TranscriptView") == {"tui/widgets/transcript.py"}
    assert _class_owners("RunObserver") == {"tui/observer.py"}


def test_rich_is_not_imported_outside_the_textual_boundary():
    owners = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if any(
            line.lstrip().startswith(("from rich", "import rich"))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert all(owner.startswith("tui/") for owner in owners)


def test_clipboard_stays_truthful_and_out_of_durable_owners():
    clipboard = (PACKAGE / "tui" / "clipboard.py").read_text(encoding="utf-8")
    storage = _python_text(PACKAGE / "storage")
    loop = _python_text(PACKAGE / "loop")
    assert _class_owners("ClipboardResult") == {"tui/clipboard.py"}
    assert "OSC 52" not in storage + loop
    tree = ast.parse(clipboard)
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
    assert "subprocess" not in top_level_imports
    pbcopy = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "copy_with_pbcopy"
    )
    assert any(
        isinstance(node, ast.Import)
        and any(alias.name == "subprocess" for alias in node.names)
        for node in ast.walk(pbcopy)
    )


def test_streaming_keeps_partial_state_disposable_and_provider_neutral():
    loop = (PACKAGE / "loop" / "driver.py").read_text(encoding="utf-8")
    app = (PACKAGE / "tui" / "app.py").read_text(encoding="utf-8")
    observation = (PACKAGE / "observation.py").read_text(encoding="utf-8")
    storage = _python_text(PACKAGE / "storage")

    assert "ModelStreamCompleted" in loop
    assert "ModelTextDelta" in loop
    assert "stream_model_calls" in loop
    assert "assistant.partial" in app
    assert "MODEL_TEXT_DELTA" in observation
    assert "assistant.partial" not in storage
    assert "MODEL_TEXT_DELTA" not in storage
    for provider in ("openai", "anthropic", "gemini", "grok", "ollama"):
        assert provider not in loop.lower()

    top_level_imports = _imports(PACKAGE / "tui" / "app.py")
    assert not any(
        module == sdk or module.startswith(f"{sdk}.")
        for module in top_level_imports
        for sdk in ("openai", "anthropic", "google", "google.genai")
    )


def test_native_stream_grammars_end_inside_provider_adapters():
    provider_root = PACKAGE / "llm" / "providers"
    owners = {
        "response.output_text.delta": provider_root / "openai.py",
        "content_block_delta": provider_root / "anthropic.py",
        "generate_content_stream": provider_root / "gemini.py",
        "stream_options": provider_root / "openai_compatible.py",
    }
    generic_runtime = "\n".join(
        (
            (PACKAGE / "loop" / "driver.py").read_text(encoding="utf-8"),
            (PACKAGE / "llm" / "routing.py").read_text(encoding="utf-8"),
            (PACKAGE / "observation.py").read_text(encoding="utf-8"),
            _python_text(PACKAGE / "storage"),
        )
    )

    for native_marker, owner in owners.items():
        assert native_marker in owner.read_text(encoding="utf-8")
        assert native_marker not in generic_runtime

    for specialization in ("grok.py", "ollama.py"):
        text = (provider_root / specialization).read_text(encoding="utf-8")
        assert "OpenAICompatibleProvider" in text
        assert "async def stream(" not in text


def test_schema_multi_selector_has_no_data_runtime_or_persisted_state_owner():
    selector_path = PACKAGE / "tui" / "screens" / "selection.py"
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
        "executors",
        "loop",
        "postgresql",
        "storage",
    )
    assert not any(
        any(fragment in module.split(".") for fragment in forbidden_fragments)
        for module in imported_modules
    )
    assert _class_owners("SelectionScreen") == {"tui/screens/selection.py"}

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
    assert "readiness_state" not in storage.lower()


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
    assert "catalog_declarations(identity.id, data_view)" in embedded
    assert "catalog_service = CatalogService(store, store)" in embedded
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


def test_sqlite_journal_and_codecs_have_one_append_only_storage_owner():
    pragma_owners = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "PRAGMA user_version" in path.read_text(encoding="utf-8")
    }
    assert pragma_owners == {"storage/sqlite_migrations/preledger.py"}
    migration_files = {
        path.name for path in (PACKAGE / "storage" / "sqlite_migrations").glob("*.py")
    }
    assert migration_files == {
        "__init__.py",
        "baseline.py",
        "database_write_receipts.py",
        "generalized_postgresql_updates.py",
        "mcp_server_bindings.py",
        "models.py",
        "postgresql_write_admission.py",
        "preledger.py",
        "runner.py",
        "scoped_source_permissions.py",
    }
    assert _class_owners("SQLiteStateStore") == {"storage/sqlite.py"}

    candidates = [PACKAGE / "loop", PACKAGE / "hosting"]
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
        "migration framework",
        "schema_version",
        "schema-version",
        "user_version",
        "sqlite_migrations",
        "sqlite_codecs",
    ):
        assert term not in text

    assert (PACKAGE / "storage" / "sqlite_migrations").is_dir()
    assert (PACKAGE / "storage" / "sqlite_codecs").is_dir()
    assert not (PACKAGE / "migrations").exists()
    production = _python_text(PACKAGE)
    for obsolete in (
        "STATE_FORMAT_VERSION",
        "_UNVERSIONED_STATE_FORMAT",
        "_StateMigration",
        "_STATE_MIGRATIONS",
        "_state_migration_path",
        "_unversioned_state_format",
        "_migrate_existing_state",
        "_migrate_v1_to_v2",
        "_migrate_v2_to_v3",
        "_require_current_source_records",
        "_UNVERSIONED_STATE_SCHEMAS",
        "_RECORD_TYPES",
        "_ENUM_TYPES",
        "def _pack(",
        "def _unpack(",
        "def _dumps(",
        "def _loads(",
    ):
        assert obsolete not in production

    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).as_posix()
        text = path.read_text(encoding="utf-8")
        if not relative.startswith("storage/"):
            assert "sqlite_migrations" not in text
            assert "sqlite_codecs" not in text


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
            "database_write_receipts",
            "learning_candidates",
            "mcp_server_bindings",
            "messages",
            "metadata",
            "postgresql_update_scopes",
            "runs",
            "semantic_annotations",
            "snapshots",
            "source_read_scopes",
            "sources",
            "state_migrations",
            "syncs",
        }
        assert columns == {
            "database_write_receipts": (
                "agent_id",
                "id",
                "run_id",
                "call_id",
                "data",
            ),
            "learning_candidates": ("agent_id", "id", "data"),
            "messages": ("run_id", "position", "data"),
            "metadata": ("key", "data"),
            "mcp_server_bindings": ("agent_id", "binding_id", "data"),
            "postgresql_update_scopes": (
                "agent_id",
                "source_id",
                "resource_id",
                "authorization_fingerprint",
                "data",
            ),
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
            "source_read_scopes": ("agent_id", "source_id", "data"),
            "sources": ("agent_id", "id", "data"),
            "state_migrations": ("ordinal", "migration_id", "checksum"),
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


def test_registry_and_common_runtime_keep_executor_resolution_ownership():
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
    assert resolution_owners == {"capabilities.py", "capability_runtime.py"}
    assert resolved_executor_callers == {"capability_runtime.py"}


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


def test_exact_tabular_extends_existing_adapter_capability_and_renderer_owners():
    assert _class_owners("ExactCsvRenderer") == {"artifacts/renderers.py"}
    assert _class_owners("ExactXlsxRenderer") == {"artifacts/renderers.py"}
    for adapter, class_name in (
        ("sqlite_query.py", "SQLiteQueryBackend"),
        ("postgresql_query.py", "PostgreSQLQueryBackend"),
    ):
        methods = _class_methods(PACKAGE / "adapters" / adapter, class_name)
        assert "execute_exact_tabular" in methods
        assert "execute_exact_csv" not in methods
        tree = ast.parse((PACKAGE / "adapters" / adapter).read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "execute_exact_tabular"
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
    package_text = _python_text(PACKAGE)
    for obsolete in (
        "ExactCsvExportBackend",
        "ExactCsvExportResult",
        "execute_exact_csv",
        "_run_exact_csv",
        "_execute_exact_csv",
        "SQLITE_CSV_EXPORT_CAPABILITY_ID",
        "POSTGRESQL_CSV_EXPORT_CAPABILITY_ID",
    ):
        assert obsolete not in package_text


def test_exact_tabular_tool_arguments_contain_query_selection_but_never_rows_or_bytes():
    from daita.domains.data.export_capabilities import (
        POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
        SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
        artifact_capability_declarations,
    )

    declarations = artifact_capability_declarations()
    for capability in declarations.capabilities:
        if capability.id not in {
            SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
            POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
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


def test_phase_three_xlsx_dependencies_are_scoped_and_integrations_remain_lazy():
    import tomllib

    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]
    assert "XlsxWriter>=3.2.5,<4.0.0" in project["dependencies"]
    assert "openpyxl>=3.1.0,<4.0.0" in project["optional-dependencies"]["dev"]
    assert "types-openpyxl>=3.1.0,<4.0.0" in project["optional-dependencies"]["dev"]
    assert all("openpyxl" not in item.casefold() for item in project["dependencies"])
    assert all("pandas" not in item.casefold() for item in project["dependencies"])
    artifact_text = _python_text(PACKAGE / "artifacts")
    assert "openpyxl" not in artifact_text
    assert "xlsxwriter" in artifact_text
    renderer = PACKAGE / "artifacts" / "renderers.py"
    top_level_imports = _imports(renderer)
    assert "xlsxwriter" not in top_level_imports
    assert "ExactXlsxRenderer" not in _python_text(PACKAGE / "loop")
    assert "ArtifactRendererRegistry" not in artifact_text
