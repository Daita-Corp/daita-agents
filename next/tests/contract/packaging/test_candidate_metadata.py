from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib

NEXT_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = NEXT_ROOT / "pyproject.toml"
SOURCE_ROOT = NEXT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "daita"

OPENAI = ("openai>=1.99.9",)
ANTHROPIC = ("anthropic>=0.116.0",)
GOOGLE = ("google-genai>=1.73.1",)
SQLITE = ("sqlglot>=25.0.0",)
POSTGRESQL = ("asyncpg>=0.30.0", "sqlglot>=25.0.0")
KEYCHAIN = ("keyring>=25.0.0",)
LLM_ALL = (*OPENAI, *ANTHROPIC, *GOOGLE)
DATA = ("asyncpg>=0.30.0", "sqlglot>=25.0.0")
RECOMMENDED = (*OPENAI, *SQLITE, *KEYCHAIN)
COMPLETE = (*LLM_ALL, *DATA, *KEYCHAIN)
DEV = (
    "black>=26.1.0",
    "build>=1.2.2",
    "mypy>=1.15.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.26.0",
    "wheel>=0.45.0",
)

EXPECTED_EXTRAS = {
    "all": COMPLETE,
    "anthropic": ANTHROPIC,
    "cli": (),
    "complete": COMPLETE,
    "data": DATA,
    "dev": DEV,
    "google": GOOGLE,
    "keychain": KEYCHAIN,
    "llm-all": LLM_ALL,
    "memory": (),
    "openai": OPENAI,
    "postgresql": POSTGRESQL,
    "production": (),
    "recommended": RECOMMENDED,
    "sqlite": SQLITE,
}


def _configuration() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _dependencies(
    extras: dict[str, list[str]],
    *names: str,
) -> frozenset[str]:
    return frozenset(dependency for name in names for dependency in extras[name])


def _distribution_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


def _literal_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                modules.append(node.args[0].value)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                modules.append(node.args[0].value)
    return tuple(modules)


def test_candidate_metadata_has_one_stable_distribution_and_cli_owner() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)

    assert project["name"] == "daita-agents"
    assert project["version"] == "2.0.0a0"
    assert project["requires-python"] == ">=3.11"
    assert project["dependencies"] == []
    assert project["scripts"] == {"daita": "daita.cli:main"}
    assert project["license"] == "Apache-2.0"

    python_classifiers = {
        classifier
        for classifier in project["classifiers"]
        if classifier.startswith("Programming Language :: Python :: 3.")
    }
    assert python_classifiers == {
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    }


def test_candidate_has_no_dependency_or_import_on_retired_cli_client_packages() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]
    assert isinstance(dependencies, list)
    assert isinstance(extras, dict)

    dependency_names = {_distribution_name(requirement) for requirement in dependencies}
    dependency_names.update(
        _distribution_name(requirement)
        for requirements in extras.values()
        for requirement in requirements
    )

    forbidden_distributions = {"daita-cli", "daita-client"}
    assert dependency_names.isdisjoint(forbidden_distributions)

    imported_roots = {
        module.split(".", 1)[0]
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for module in _literal_imports(path)
    }
    assert imported_roots.isdisjoint({"daita_cli", "daita_client"})


def test_candidate_has_no_legacy_local_http_runtime_fallback() -> None:
    forbidden_module_names = {"local_agent_server.py", "local_server_client.py"}
    source_paths = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert not {path.name for path in source_paths} & forbidden_module_names

    candidate_text = "\n".join(
        [PYPROJECT.read_text(encoding="utf-8")]
        + [path.read_text(encoding="utf-8") for path in source_paths]
    ).casefold()
    for legacy_sentinel in (
        "8123",
        "default_local_server_url",
        "local_agent_server",
        "local_server_client",
        "serve_dev_server",
        "sqliteruntimestore",
    ):
        assert legacy_sentinel not in candidate_text


def test_distribution_readme_has_no_links_to_source_only_local_files() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)
    readme_name = project["readme"]
    assert readme_name == "README.md"
    assert isinstance(readme_name, str)

    readme = (NEXT_ROOT / readme_name).read_text(encoding="utf-8")
    local_links = sorted(
        target
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
        if not target.startswith("#")
        and re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) is None
    )

    assert local_links == []


def test_optional_extra_names_and_dependencies_are_exactly_the_supported_set() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)

    assert set(extras) == set(EXPECTED_EXTRAS)
    assert {
        name: tuple(dependencies) for name, dependencies in extras.items()
    } == EXPECTED_EXTRAS

    runtime_dependencies = {
        _distribution_name(dependency)
        for name, dependencies in extras.items()
        if name != "dev"
        for dependency in dependencies
    }
    assert runtime_dependencies == {
        "anthropic",
        "asyncpg",
        "google-genai",
        "keyring",
        "openai",
        "sqlglot",
    }


def test_bundle_extras_are_deduplicated_unions_of_supported_atomic_extras() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)

    assert _dependencies(extras, "llm-all") == _dependencies(
        extras,
        "openai",
        "anthropic",
        "google",
    )
    assert _dependencies(extras, "data") == _dependencies(
        extras,
        "sqlite",
        "postgresql",
    )
    assert _dependencies(extras, "recommended") == _dependencies(
        extras,
        "openai",
        "sqlite",
        "keychain",
    )
    assert _dependencies(extras, "complete") == _dependencies(
        extras,
        "llm-all",
        "data",
        "keychain",
    )
    assert extras["all"] == extras["complete"]
    assert extras["cli"] == []
    assert extras["memory"] == []
    assert extras["production"] == []
    assert all(
        len(dependencies) == len(set(dependencies)) for dependencies in extras.values()
    )


def test_deferred_v1_integrations_are_not_advertised_by_any_v2_extra() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)

    deferred_extra_names = {
        "api-server",
        "aws",
        "azure",
        "bigquery",
        "chromadb",
        "cloud",
        "data-quality",
        "databases",
        "elasticsearch",
        "exa",
        "gcp",
        "github",
        "google-drive",
        "lineage",
        "mcp",
        "mongodb",
        "mysql",
        "neo4j",
        "opensearch",
        "otlp",
        "pinecone",
        "qdrant",
        "redis",
        "sentence-transformers",
        "slack",
        "snowflake",
        "transformers",
        "vectordb",
        "voyage",
        "web",
        "websearch",
    }
    assert deferred_extra_names.isdisjoint(extras)
    assert all(
        "opentelemetry" not in dependency.lower()
        for dependencies in extras.values()
        for dependency in dependencies
    )


def test_setuptools_discovery_has_a_source_only_distribution_allowlist() -> None:
    configuration = _configuration()
    tool_configuration = configuration["tool"]
    assert isinstance(tool_configuration, dict)
    setuptools = tool_configuration["setuptools"]
    assert isinstance(setuptools, dict)

    assert setuptools["package-dir"] == {"": "src"}
    assert setuptools["packages"]["find"] == {
        "where": ["src"],
        "include": ["daita*"],
    }
    assert "package-data" not in setuptools
    assert "data-files" not in setuptools

    source_files = tuple(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert source_files
    assert all(path.suffix == ".py" for path in source_files)
    assert all(not path.is_symlink() for path in source_files)
    assert all(
        path.relative_to(SOURCE_ROOT).parts[0] == "daita" for path in source_files
    )
    assert all(
        (path.parent / "__init__.py").is_file()
        for path in source_files
        if path.name != "__init__.py"
    )
