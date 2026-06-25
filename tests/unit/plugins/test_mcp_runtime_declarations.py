from daita.plugins import ExtensionRegistry
from daita.plugins.base import PluginContext, ServiceRegistry
from daita.plugins.mcp import MCPServer, MCPTool
from daita.runtime import InMemoryRuntimeStore, RuntimeKernel


class FakeMCPServer(MCPServer):
    def __init__(self):
        super().__init__("fake", server_name="calculator")
        self.calls = []

    async def connect(self) -> None:
        self._tools = [
            MCPTool(
                name="add",
                description="Add two numbers.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            )
        ]
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def _call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return arguments["a"] + arguments["b"]


async def test_mcp_setup_registers_extension_first_tool_declarations():
    registry = ExtensionRegistry()
    server = FakeMCPServer()
    registry.register(server)

    await registry.setup_all(
        PluginContext(
            runtime_id="agent-1",
            runtime_kind="agent",
            services=ServiceRegistry({"extension_registry": registry}),
        ),
    )

    owner = registry.get_tool_view_owner("calculator_add")
    capability = registry.get_capability("mcp.calculator.add", owner=owner)

    assert owner == "mcp_calculator"
    assert capability.owner == "mcp_calculator"
    assert capability.executor == "mcp_calculator.add"
    assert registry.get_executor("mcp_calculator.add").tool_name == "add"
    assert registry.get_evidence_schema("mcp.tool.result", owner=owner)
    assert set(registry.plugin_ids) == {"mcp_server_calculator", "mcp_calculator"}


async def test_mcp_tool_execution_flows_through_runtime_kernel():
    registry = ExtensionRegistry()
    server = FakeMCPServer()
    registry.register(server)
    await registry.setup_plugin(
        server.manifest.id,
        PluginContext(
            runtime_id="agent-1",
            runtime_kind="agent",
            services=ServiceRegistry({"extension_registry": registry}),
        ),
    )
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="agent-1",
        runtime_kind="chat",
        extension_registry=registry,
        runtime_store=store,
    )
    operation = await kernel.create_operation(
        operation_type="chat.run",
        request={"prompt": "add"},
    )

    result = await kernel.execute_capability(
        "mcp.calculator.add",
        owner="mcp_calculator",
        operation_id=operation.id,
        input={"a": 2, "b": 5},
        context={"tool_owner": "mcp_calculator"},
    )

    assert server.calls == [("add", {"a": 2, "b": 5})]
    assert result.evidence[0].kind == "mcp.tool.result"
    assert result.evidence[0].owner == "mcp_calculator"
    assert result.evidence[0].payload["result"] == 7


def test_mcp_direct_registry_routing_is_removed():
    import daita.plugins.mcp as mcp

    assert not hasattr(mcp, "MCPToolRegistry")
    assert not hasattr(MCPServer, "call_tool")
