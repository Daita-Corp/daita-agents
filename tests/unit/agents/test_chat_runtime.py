from __future__ import annotations

from dataclasses import replace

from daita.agents.agent import Agent
from daita.agents.chat.runtime import ChatRunResult
from daita.core.exceptions import SkillError
from daita.plugins import ExtensionRegistry, PluginKind, PluginManifest
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    Operation,
    OperationStatus,
    RiskLevel,
    RuntimeEventType,
    Task,
    TaskExecutionResult,
    TaskStatus,
    ToolView,
)
from daita.skills import Skill, SkillDiscovery

from tests.conftest import SequentialMockLLM


class CountingExecutor:
    id = "chat_test.lookup"
    capability_ids = frozenset({"chat.lookup"})

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, task, operation, context):
        self.calls += 1
        return [
            Evidence(
                kind="chat.lookup.result",
                owner="chat_test",
                payload={"value": task.input.get("query")},
            )
        ]


class ChatToolPlugin:
    manifest = PluginManifest(
        id="chat_test",
        display_name="Chat Test",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self, *, runtime_only: bool = False) -> None:
        self.executor = CountingExecutor()
        self.runtime_only = runtime_only

    def declare_capabilities(self):
        return (
            Capability(
                id="chat.lookup",
                owner="chat_test",
                description="Lookup a value.",
                domains=frozenset({"chat"}),
                operation_types=frozenset({"chat.lookup"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"chat.lookup.result"}),
                executor=self.executor.id,
                runtime_only=self.runtime_only,
                side_effecting=False,
                replay_safe=True,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def get_tool_views(self):
        if self.runtime_only:
            return ()
        return (
            ToolView(
                name="lookup_value",
                capability_id="chat.lookup",
                description="Lookup a value.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        )


async def test_agent_run_delegates_to_chat_runtime():
    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            return ChatRunResult(
                text="from runtime",
                operation_id="chat-op",
                iterations=1,
                tool_calls=(),
                diagnostics={},
                token_usage={},
                cost=0.0,
            )

    agent = Agent(name="Facade", llm_provider=SequentialMockLLM(["unused"]))
    fake = FakeRuntime()
    agent.runtime = fake

    result = await agent.run("hello")

    assert result == "from runtime"
    assert fake.calls[0]["prompt"] == "hello"


async def test_agent_run_unknown_explicit_skill_selection_raises():
    agent = Agent(
        name="SkillChat",
        llm_provider=SequentialMockLLM(["unused"]),
        skills=[Skill(name="finance", instructions="Use finance context.")],
    )

    try:
        await agent.run("hello", skills=["finacne"])
    except SkillError as exc:
        assert "Unknown skill selection(s): finacne" in str(exc)
    else:
        raise AssertionError("expected unknown skill selection to raise")


async def test_chat_tool_call_uses_kernel_execute_capability_not_direct_executor():
    plugin = ChatToolPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    llm = SequentialMockLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "lookup_value",
                        "arguments": {"query": "alpha"},
                    }
                ],
            },
            "alpha found",
        ]
    )
    agent = Agent(name="KernelChat", llm_provider=llm, extension_registry=registry)
    calls = []
    original = agent.runtime_kernel.execute_capability

    async def spy_execute_capability(capability_id, **kwargs):
        calls.append((capability_id, kwargs))
        operation = await agent.runtime_store.load_operation(kwargs["operation_id"])
        capability = agent.extension_registry.get_capability(
            capability_id,
            owner=kwargs["owner"],
        )
        task = Task(
            id="spy-task",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input=dict(kwargs["input"]),
            status=TaskStatus.SUCCEEDED,
            metadata={"owner": capability.owner},
        )
        return TaskExecutionResult(
            operation=operation,
            task=task,
            capability=capability,
            evidence=(
                Evidence(
                    kind="chat.lookup.result",
                    owner=capability.owner,
                    operation_id=operation.id,
                    task_id=task.id,
                    payload={"value": "alpha"},
                ),
            ),
        )

    agent.runtime_kernel.execute_capability = spy_execute_capability
    try:
        result = await agent.run("lookup", detailed=True)
    finally:
        agent.runtime_kernel.execute_capability = original

    assert result["result"] == "alpha found"
    assert calls[0][0] == "chat.lookup"
    assert calls[0][1]["operation_type"] == "chat.tool_call"
    assert calls[0][1]["context"]["tool_view"]["name"] == "lookup_value"
    assert plugin.executor.calls == 0


async def test_chat_tool_call_records_runtime_correlated_observability():
    plugin = ChatToolPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    llm = SequentialMockLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "lookup_value",
                        "arguments": {"query": "alpha"},
                    }
                ],
            },
            "alpha found",
        ]
    )
    agent = Agent(name="ObservableChat", llm_provider=llm, extension_registry=registry)

    result = await agent.run("lookup", detailed=True)
    snapshot = await agent.runtime_store.inspect_operation(result["operation_id"])

    assert plugin.executor.calls == 1
    assert snapshot is not None
    event_types = [event.type for event in snapshot.events]
    assert RuntimeEventType.LLM_REQUESTED in event_types
    assert RuntimeEventType.LLM_COMPLETED in event_types
    assert RuntimeEventType.EXECUTOR_STARTED in event_types
    assert RuntimeEventType.EVIDENCE_ACCEPTED in event_types
    assert RuntimeEventType.EXECUTOR_COMPLETED in event_types
    evidence = snapshot.evidence[0]
    evidence_event = next(
        event
        for event in snapshot.events
        if event.type is RuntimeEventType.EVIDENCE_ACCEPTED
    )
    assert evidence_event.runtime_id == agent.agent_id
    assert evidence_event.runtime_kind == "chat"
    assert evidence_event.operation_id == result["operation_id"]
    assert evidence_event.task_id == evidence.task_id
    assert evidence_event.capability_id == "chat.lookup"
    assert evidence_event.executor_id == "chat_test.lookup"
    assert evidence_event.plugin_id == "chat_test"
    assert evidence_event.evidence_id == evidence.id


async def test_agent_execute_capability_uses_runtime_kernel_execute_task():
    plugin = ChatToolPlugin()
    agent = Agent(name="DirectCapability", llm_provider=SequentialMockLLM(["unused"]))
    agent.add_plugin(plugin)
    calls = []
    original = agent.runtime_kernel.execute_task

    async def spy_execute_task(task_id, *, context=None, **kwargs):
        calls.append((task_id, context))
        task = await agent.runtime_store.load_task(task_id)
        operation = await agent.runtime_store.load_operation(task.operation_id)
        capability = agent.extension_registry.get_capability(
            task.capability_id,
            owner=task.metadata["owner"],
        )
        return TaskExecutionResult(
            operation=operation,
            task=replace(task, status=TaskStatus.SUCCEEDED),
            capability=capability,
            evidence=(
                Evidence(
                    kind="chat.lookup.result",
                    owner=capability.owner,
                    operation_id=operation.id,
                    task_id=task.id,
                    payload={"value": task.input["query"]},
                ),
            ),
        )

    agent.runtime_kernel.execute_task = spy_execute_task
    try:
        result = await agent.execute_capability(
            "chat.lookup",
            {"query": "beta"},
            owner="chat_test",
        )
    finally:
        agent.runtime_kernel.execute_task = original

    assert calls
    assert result["evidence"][0]["payload"]["value"] == "beta"
    assert plugin.executor.calls == 0


async def test_model_visible_tool_views_map_to_capabilities():
    plugin = ChatToolPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    llm = SequentialMockLLM(["final"])
    agent = Agent(name="Projection", llm_provider=llm, extension_registry=registry)

    await agent.run("what tools?")

    tools = llm.call_history[0]["tools"]
    assert [tool["function"]["name"] for tool in tools] == ["lookup_value"]
    spec = agent.runtime._prepare_model_tools
    projected = await spec(None)
    assert projected[0].capability_id == "chat.lookup"
    assert projected[0].owner == "chat_test"


async def test_runtime_only_capabilities_are_not_model_visible():
    plugin = ChatToolPlugin(runtime_only=True)
    registry = ExtensionRegistry()
    registry.register(plugin)
    llm = SequentialMockLLM(["final"])
    agent = Agent(name="Projection", llm_provider=llm, extension_registry=registry)

    await agent.run("what tools?")

    assert llm.call_history[0]["tools"] is None


async def test_local_agent_tool_is_wrapped_as_runtime_capability():
    async def add(args):
        return args["a"] + args["b"]

    from daita.core.tools import LocalTool

    tool = LocalTool(
        name="add",
        description="Add values.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=add,
        side_effecting=False,
        replay_safe=True,
    )
    llm = SequentialMockLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "name": "add", "arguments": {"a": 2, "b": 5}}
                ],
            },
            "The answer is 7.",
        ]
    )
    agent = Agent(name="LocalTool", llm_provider=llm, tools=[tool])

    result = await agent.run("add", detailed=True)

    tasks = await agent.runtime_store.list_tasks(result["operation_id"])
    assert result["tool_calls"][0]["result"] == 7
    assert tasks[0].capability_id == "agent.local.add"
    assert tasks[0].metadata["tool_view"] == "add"


async def test_tool_decorator_executes_through_runtime_kernel():
    from daita.core.tools import tool

    calls = {"handler": 0, "kernel": []}

    @tool(id="math.add", output_evidence="math.add.result", side_effecting=False)
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls["handler"] += 1
        return a + b

    llm = SequentialMockLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "name": "add", "arguments": {"a": 2, "b": 5}}
                ],
            },
            "The answer is 7.",
        ]
    )
    agent = Agent(name="DecoratedTool", llm_provider=llm, tools=[add])
    original = agent.runtime_kernel.execute_capability

    async def spy_execute_capability(capability_id, **kwargs):
        calls["kernel"].append((capability_id, kwargs))
        return await original(capability_id, **kwargs)

    agent.runtime_kernel.execute_capability = spy_execute_capability
    try:
        result = await agent.run("add", detailed=True)
    finally:
        agent.runtime_kernel.execute_capability = original

    assert result["tool_calls"][0]["result"] == 7
    assert calls["handler"] == 1
    assert calls["kernel"][0][0] == "math.add"
    assert calls["kernel"][0][1]["operation_type"] == "chat.tool_call"


async def test_explicit_run_skill_selection_loads_context_and_records_events():
    skill = Skill(
        name="finance",
        description="Finance analysis.",
        instructions="Full finance instructions.",
        context_mode="on_demand",
        discovery=SkillDiscovery(
            name="finance",
            description="Finance analysis.",
            context_mode="on_demand",
        ),
    )
    llm = SequentialMockLLM(["done"])
    agent = Agent(name="SkillSelect", llm_provider=llm, skills=[skill])

    result = await agent.run("hello", skills=["finance"], detailed=True)
    snapshot = await agent.runtime_store.inspect_operation(result["operation_id"])

    system = llm.call_history[0]["messages"][0]["content"]
    assert "Full finance instructions." in system
    assert snapshot is not None
    diagnostics = [event.payload.get("diagnostic") for event in snapshot.events]
    assert "skill.selected" in diagnostics
    assert "skill.context_loaded" in diagnostics


async def test_tool_backed_skill_tool_execution_records_skill_event():
    from daita.core.tools import tool

    @tool(id="report.format", output_evidence="report.format.result")
    async def format_report(title: str) -> str:
        """Format a report."""
        return f"# {title}"

    skill = Skill.with_tools(
        name="report_builder",
        instructions="Use reports.",
        tools=[format_report],
    )
    llm = SequentialMockLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "format_report",
                        "arguments": {"title": "Revenue"},
                    }
                ],
            },
            "done",
        ]
    )
    agent = Agent(name="SkillTool", llm_provider=llm, skills=[skill])

    result = await agent.run("format", detailed=True)
    snapshot = await agent.runtime_store.inspect_operation(result["operation_id"])

    assert result["tool_calls"][0]["capability_ids"] == ("report.format",)
    assert snapshot is not None
    diagnostics = [event.payload.get("diagnostic") for event in snapshot.events]
    assert "skill.tool_executed" in diagnostics
