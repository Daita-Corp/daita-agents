"""
OrchestratorPlugin for multi-agent coordination and task routing.

Provides tools for finding capable agents, routing tasks, and executing
workflows across multiple agents.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime

from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool
    from ..agents.agent import Agent

logger = logging.getLogger(__name__)


class OrchestratorPlugin(BasePlugin):
    """
    Plugin for multi-agent orchestration and coordination.

    Works standalone (with local agent registry) or with optional graph storage
    for capability lookup and performance tracking.

    Supports:
    - Agent capability registry
    - Task routing and dispatch
    - Parallel and sequential execution
    - Workflow DAG definition and execution
    - Agent performance metrics
    """

    def __init__(
        self,
        agents: Optional[Dict[str, 'Agent']] = None,
        graph_store: Optional[Any] = None,
        organization_id: Optional[int] = None,
        llm_provider: Optional[Any] = None,
        routing_model: Optional[str] = "gpt-4o-mini",
        routing_api_key: Optional[str] = None,
        enable_llm_routing: bool = True,
        routing_cache_ttl: int = 300,
        max_description_length: int = 500
    ):
        """
        Initialize OrchestratorPlugin.

        Args:
            agents: Optional dict of agent name -> Agent instance
            graph_store: Optional graph storage for capability lookup
            organization_id: Optional organization ID for multi-tenant graphs
            llm_provider: LLM provider name ('openai', 'anthropic', etc.) or BaseLLMProvider instance
            routing_model: Model to use for routing (default: 'gpt-4o-mini')
            routing_api_key: API key for routing LLM
            enable_llm_routing: Enable LLM-based routing (default: True)
            routing_cache_ttl: Cache TTL in seconds for routing decisions (default: 300)
            max_description_length: Max length for agent descriptions (default: 500)
        """
        self._agents = agents or {}
        self._graph_store = graph_store
        self._organization_id = organization_id

        # LLM routing configuration
        self._llm_provider_config = llm_provider
        self._routing_model = routing_model
        self._routing_api_key = routing_api_key
        self._enable_llm_routing = enable_llm_routing
        self._routing_cache_ttl = routing_cache_ttl
        self._max_description_length = max_description_length

        # Routing LLM instance (lazy-initialized)
        self._routing_llm = None
        self._routing_llm_initialized = False

        # Routing cache: {cache_key: {agent_id, timestamp}}
        self._routing_cache = {}

        # Capability registry (in-memory)
        self._capabilities = {}  # agent_id -> {capabilities, entity_types, performance, description}

        # Workflow definitions
        self._workflows = {}  # workflow_id -> workflow definition

        # Auto-register agents if provided
        if self._agents:
            self._auto_register_agents(self._agents)

        logger.debug(f"OrchestratorPlugin initialized (agents: {len(self._agents)}, llm_routing: {enable_llm_routing}, graph_store: {graph_store is not None})")

    def _auto_register_agents(self, agents: Dict[str, 'Agent']) -> None:
        """
        Auto-register agents using their prompts and tools as capability metadata.

        Args:
            agents: Dict of agent_id -> Agent instance
        """
        for agent_id, agent in agents.items():
            # Extract capabilities from agent's tools
            capabilities = []
            if hasattr(agent, 'tool_registry') and agent.tool_registry:
                capabilities = list(agent.tool_registry.tool_names)

            # Extract description from agent's prompt
            description = ""
            if hasattr(agent, 'prompt') and agent.prompt:
                prompt = agent.prompt
                # Handle dict prompts (extract system message)
                if isinstance(prompt, dict):
                    prompt = prompt.get('system', str(prompt))
                description = str(prompt)[:self._max_description_length]

            # Register in capabilities dict
            self._capabilities[agent_id] = {
                "capabilities": capabilities,
                "description": description,
                "entity_types": [],
                "registered_at": datetime.utcnow().isoformat(),
                "total_executions": 0,
                "successful_executions": 0,
                "total_time_ms": 0,
                "auto_registered": True
            }

        logger.info(f"Auto-registered {len(agents)} agents with LLM-based routing")

    def _setup_routing_llm(self) -> None:
        """Initialize the LLM provider for routing decisions."""
        if self._routing_llm_initialized:
            return

        self._routing_llm_initialized = True

        if not self._enable_llm_routing:
            self._routing_llm = None
            logger.debug("LLM routing disabled")
            return

        try:
            from ..llm.factory import create_llm_provider
            from ..config.settings import settings

            # If LLM provider is already an instance, use it
            if hasattr(self._llm_provider_config, 'generate'):
                self._routing_llm = self._llm_provider_config
                logger.debug("Using provided LLM instance for routing")
                return

            # Otherwise create a new provider
            provider = self._llm_provider_config or 'openai'
            model = self._routing_model or 'gpt-4o-mini'

            # Get API key from settings if not provided
            api_key = self._routing_api_key
            if not api_key:
                api_key = settings.get_llm_api_key(provider)

            if not api_key:
                logger.warning(
                    f"No API key available for {provider}. "
                    "LLM routing disabled, falling back to capability matching."
                )
                self._routing_llm = None
                return

            self._routing_llm = create_llm_provider(
                provider=provider,
                model=model,
                api_key=api_key,
                agent_id=f"orchestrator_router_{id(self)}"
            )
            logger.debug(f"Initialized routing LLM: {provider}/{model}")

        except Exception as e:
            logger.warning(f"Failed to initialize routing LLM: {e}. Falling back to capability matching.")
            self._routing_llm = None

    def _build_agent_routing_tools(self) -> List['AgentTool']:
        """
        Convert registered agents to AgentTool objects for routing.

        Returns:
            List of AgentTool objects (provider handles format conversion automatically)
        """
        from ..core.tools import AgentTool

        tools = []
        for agent_id, agent in self._agents.items():
            # Get description from capabilities registry
            agent_info = self._capabilities.get(agent_id, {})
            description = agent_info.get("description", f"Agent: {agent_id}")

            # Add capabilities to description if available
            capabilities = agent_info.get("capabilities", [])
            if capabilities:
                cap_list = ", ".join(capabilities[:5])  # Limit to first 5
                description = f"{description}\n\nCapabilities: {cap_list}"

            # Create AgentTool object (provider will convert to correct format)
            tool = AgentTool(
                name=f"route_to_{agent_id}",
                description=description,
                parameters={
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why this agent is suitable for the task"
                        }
                    },
                    "required": []
                },
                handler=lambda **kwargs: None,  # Dummy handler (routing tools aren't executed)
                category="routing",
                source="orchestrator",
                plugin_name="Orchestrator"
            )
            tools.append(tool)

        return tools

    async def _route_task_via_llm(self, task: str) -> str:
        """
        Use LLM to determine the best agent for a task.

        Args:
            task: Task description

        Returns:
            Agent ID string

        Raises:
            RoutingError: If routing fails
        """
        from ..core.exceptions import RoutingError

        # Check we have agents
        if not self._agents:
            raise RoutingError(
                "No agents registered in orchestrator",
                task=task,
                available_agents=[]
            )

        # Single agent optimization
        if len(self._agents) == 1:
            return list(self._agents.keys())[0]

        # Check cache
        cache_key = f"task:{task[:100]}"  # Use first 100 chars as key
        if cache_key in self._routing_cache:
            cached = self._routing_cache[cache_key]
            if time.time() - cached['timestamp'] < self._routing_cache_ttl:
                logger.debug(f"Cache hit for routing: {task[:50]}... -> {cached['agent_id']}")
                return cached['agent_id']

        # Lazy init routing LLM
        if not self._routing_llm_initialized:
            self._setup_routing_llm()

        # Fall back to legacy if LLM unavailable
        if not self._routing_llm:
            logger.debug("LLM unavailable, using legacy routing")
            return await self._route_task_legacy(task)

        try:
            # Build routing tools
            routing_tools = self._build_agent_routing_tools()

            # System prompt for routing
            system_prompt = """You are a task router. Given a task, select the most appropriate agent to handle it.

Available agents are provided as tools. Call exactly ONE tool - the one for the agent best suited for the task.
Consider the agent's description, capabilities, and the nature of the task.

IMPORTANT: You must call a tool. Do not respond with text only."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Route this task to the best agent: {task}"}
            ]

            # Use proper generate() API - provider handles tool format conversion automatically
            response = await self._routing_llm.generate(
                messages=messages,
                tools=routing_tools,  # AgentTool objects, converted by provider
                temperature=0.0  # Deterministic routing
            )

            # Extract agent_id from tool call
            # Response can be either a dict (with tool_calls) or a string (text response)
            if isinstance(response, dict) and "tool_calls" in response:
                tool_calls = response["tool_calls"]
                if tool_calls:
                    tool_call = tool_calls[0]
                    tool_name = tool_call.get("name", "")
                    if tool_name.startswith("route_to_"):
                        agent_id = tool_name[len("route_to_"):]

                        # Cache result
                        self._routing_cache[cache_key] = {
                            'agent_id': agent_id,
                            'timestamp': time.time(),
                            'reasoning': tool_call.get("arguments", {}).get("reasoning", "")
                        }

                        logger.debug(f"LLM routing: {task[:50]}... -> {agent_id}")
                        return agent_id

            # LLM didn't call a tool (returned text or empty) - fall back
            logger.warning(f"LLM failed to call routing tool for task: {task[:100]}")
            return await self._route_task_legacy(task)

        except Exception as e:
            logger.warning(f"LLM routing failed: {e}. Falling back to legacy routing.")
            return await self._route_task_legacy(task)

    async def _route_task_legacy(self, task: str) -> str:
        """
        Fallback routing using simple keyword matching.

        Args:
            task: Task description

        Returns:
            Agent ID string

        Raises:
            RoutingError: If no suitable agent found
        """
        from ..core.exceptions import RoutingError

        if not self._agents:
            raise RoutingError(
                "No agents available for routing",
                task=task,
                available_agents=[]
            )

        # Score agents based on keyword matching
        task_lower = task.lower()
        scores = {}

        for agent_id, agent in self._agents.items():
            score = 0
            agent_info = self._capabilities.get(agent_id, {})

            # Match agent name
            agent_name_words = agent_id.lower().replace('_', ' ').split()
            for word in agent_name_words:
                if word in task_lower:
                    score += 10

            # Match capabilities
            capabilities = agent_info.get("capabilities", [])
            for cap in capabilities:
                cap_words = cap.lower().replace('_', ' ').split()
                for word in cap_words:
                    if word in task_lower:
                        score += 5

            # Match description
            description = agent_info.get("description", "").lower()
            task_words = task_lower.split()
            for word in task_words:
                if len(word) > 3 and word in description:
                    score += 3

            if score > 0:
                scores[agent_id] = score

        if not scores:
            # No match found - use first agent with warning
            first_agent = list(self._agents.keys())[0]
            logger.warning(f"No keyword matches for task: {task[:100]}. Using first agent: {first_agent}")
            return first_agent

        # Return highest scoring agent
        best_agent = max(scores, key=scores.get)
        logger.debug(f"Legacy routing: {task[:50]}... -> {best_agent} (score: {scores[best_agent]})")
        return best_agent

    def get_tools(self) -> List['AgentTool']:
        """
        Expose orchestration operations as agent tools.

        Returns:
            List of AgentTool instances for orchestration operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="find_agent",
                description="Find the best agent for a task using natural language description. Returns the agent ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language description of the task (e.g., 'analyze Q4 revenue data', 'generate executive summary')"
                        }
                    },
                    "required": ["task"]
                },
                handler=self._tool_find_agent,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=30
            ),
            AgentTool(
                name="get_performance",
                description="Get performance metrics for an agent (execution count, avg time, success rate)",
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Agent ID or name"
                        },
                        "time_range_days": {
                            "type": "integer",
                            "description": "Look back period in days (default: 30)"
                        }
                    },
                    "required": ["agent_id"]
                },
                handler=self._tool_get_performance,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=30
            ),
            AgentTool(
                name="get_capabilities",
                description="Get an agent's specializations and capabilities",
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Agent ID or name"
                        }
                    },
                    "required": ["agent_id"]
                },
                handler=self._tool_get_capabilities,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=30
            ),
            AgentTool(
                name="route_task",
                description="Route a task to the best agent and execute it",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task description or prompt"
                        },
                        "required_capabilities": {
                            "type": "array",
                            "description": "Optional required capabilities filter",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["task"]
                },
                handler=self._tool_route_task,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=300
            ),
            AgentTool(
                name="run_parallel",
                description="Execute multiple tasks across agents in parallel. Tasks can be strings (auto-routed) or objects with optional agent_id",
                parameters={
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "description": "List of tasks - each can be a string (auto-routed) or {\"task\": \"...\", \"agent_id\": \"...\"} (explicit)",
                            "items": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "task": {"type": "string", "description": "Task description"},
                                            "agent_id": {"type": "string", "description": "Optional explicit agent ID (auto-routed if omitted)"}
                                        },
                                        "required": ["task"]
                                    }
                                ]
                            }
                        }
                    },
                    "required": ["tasks"]
                },
                handler=self._tool_run_parallel,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=300
            ),
            AgentTool(
                name="run_sequential",
                description="Execute tasks in sequence, passing results between agents. Tasks can be strings (auto-routed) or objects with optional agent_id",
                parameters={
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "description": "List of tasks - each can be a string (auto-routed) or {\"task\": \"...\", \"agent_id\": \"...\"} (explicit)",
                            "items": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "task": {"type": "string", "description": "Task description"},
                                            "agent_id": {"type": "string", "description": "Optional explicit agent ID (auto-routed if omitted)"}
                                        },
                                        "required": ["task"]
                                    }
                                ]
                            }
                        }
                    },
                    "required": ["tasks"]
                },
                handler=self._tool_run_sequential,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=300
            ),
            AgentTool(
                name="create_workflow",
                description="Define a new workflow as a DAG of steps with dependencies. Agent ID is optional and will be auto-routed if omitted",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Workflow name"
                        },
                        "steps": {
                            "type": "array",
                            "description": "List of workflow steps",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step_id": {"type": "string", "description": "Unique step identifier"},
                                    "agent_id": {"type": "string", "description": "Optional explicit agent ID (auto-routed if omitted)"},
                                    "task": {"type": "string", "description": "Task description"},
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of step IDs this step depends on"
                                    }
                                },
                                "required": ["step_id", "task"]
                            }
                        }
                    },
                    "required": ["name", "steps"]
                },
                handler=self._tool_create_workflow,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=30
            ),
            AgentTool(
                name="run_workflow",
                description="Execute a defined workflow DAG",
                parameters={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "Workflow ID or name"
                        },
                        "input_data": {
                            "type": "object",
                            "description": "Input data for the workflow"
                        }
                    },
                    "required": ["workflow_id"]
                },
                handler=self._tool_run_workflow,
                category="orchestration",
                source="plugin",
                plugin_name="Orchestrator",
                timeout_seconds=600
            )
        ]

    def register_agent(
        self,
        agent: 'Agent',
        agent_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None
    ) -> None:
        """
        Register an agent in the orchestrator.

        Args:
            agent: Agent instance
            agent_id: Optional agent ID (defaults to agent.name)
            capabilities: List of capabilities
            entity_types: List of entity types the agent works with
        """
        agent_id = agent_id or agent.name

        self._agents[agent_id] = agent
        self._capabilities[agent_id] = {
            "capabilities": capabilities or [],
            "entity_types": entity_types or [],
            "registered_at": datetime.utcnow().isoformat(),
            "total_executions": 0,
            "successful_executions": 0,
            "total_time_ms": 0
        }

        logger.info(f"Registered agent: {agent_id}")

    async def _tool_find_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_agent"""
        task = args.get("task", "")

        try:
            agent_id = await self.find_agent(task)
            return {
                "success": True,
                "agent_id": agent_id,
                "message": f"Selected agent: {agent_id}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def find_agent(self, task: str) -> str:
        """
        Find the best agent for a task using LLM-based routing.

        Args:
            task: Natural language task description

        Returns:
            Agent ID string

        Raises:
            RoutingError: If no suitable agent found
        """
        return await self._route_task_via_llm(task)

    async def _tool_get_performance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_performance"""
        agent_id = args.get("agent_id")
        time_range_days = args.get("time_range_days", 30)

        result = await self.get_performance(agent_id, time_range_days)
        return result

    async def get_performance(
        self,
        agent_id: str,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance metrics for an agent.

        Args:
            agent_id: Agent ID
            time_range_days: Look back period

        Returns:
            Performance metrics
        """
        if agent_id not in self._capabilities:
            return {
                "success": False,
                "error": f"Agent not found: {agent_id}"
            }

        agent_info = self._capabilities[agent_id]
        total = agent_info.get("total_executions", 0)

        if total == 0:
            success_rate = 0
            avg_time = 0
        else:
            success_rate = agent_info.get("successful_executions", 0) / total * 100
            avg_time = agent_info.get("total_time_ms", 0) / total

        return {
            "success": True,
            "agent_id": agent_id,
            "total_executions": total,
            "successful_executions": agent_info.get("successful_executions", 0),
            "success_rate_percent": round(success_rate, 2),
            "avg_execution_time_ms": round(avg_time, 2),
            "time_range_days": time_range_days
        }

    async def _tool_get_capabilities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_capabilities"""
        agent_id = args.get("agent_id")

        result = await self.get_capabilities(agent_id)
        return result

    async def get_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent capabilities and specializations.

        Args:
            agent_id: Agent ID

        Returns:
            Agent capabilities
        """
        if agent_id not in self._capabilities:
            return {
                "success": False,
                "error": f"Agent not found: {agent_id}"
            }

        agent_info = self._capabilities[agent_id]

        return {
            "success": True,
            "agent_id": agent_id,
            "capabilities": agent_info.get("capabilities", []),
            "entity_types": agent_info.get("entity_types", []),
            "registered_at": agent_info.get("registered_at")
        }

    async def _normalize_task(self, task_input: Union[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Normalize task input to dict format with task and agent_id.

        Args:
            task_input: Either a string (task description) or dict with 'task' and optional 'agent_id'

        Returns:
            Dict with 'task' and 'agent_id' keys

        Raises:
            RoutingError: If routing fails
        """
        # String input - auto-route
        if isinstance(task_input, str):
            agent_id = await self._route_task_via_llm(task_input)
            return {"task": task_input, "agent_id": agent_id}

        # Dict input
        if isinstance(task_input, dict):
            # Explicit agent_id - use as-is (override)
            if "agent_id" in task_input:
                return {"task": task_input.get("task", ""), "agent_id": task_input["agent_id"]}

            # Dict without agent_id - auto-route
            task = task_input.get("task", str(task_input))
            agent_id = await self._route_task_via_llm(task)
            return {"task": task, "agent_id": agent_id}

        # Invalid input type
        raise ValueError(f"Invalid task input type: {type(task_input)}. Expected str or dict.")

    async def _tool_route_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for route_task"""
        task = args.get("task")

        result = await self.route_task(task)
        return result

    async def route_task(self, task: str) -> Dict[str, Any]:
        """
        Route a task to the best agent and execute it.

        Args:
            task: Task description

        Returns:
            Task execution result
        """
        # Find best agent using new routing
        agent_id = await self.find_agent(task)

        if agent_id not in self._agents:
            return {
                "success": False,
                "error": f"Agent found but not available: {agent_id}"
            }

        agent = self._agents[agent_id]

        # Execute task
        try:
            start_time = datetime.utcnow()

            # Check if agent has run method
            if hasattr(agent, 'run'):
                result = await agent.run(task)
            else:
                result = "Agent executed (no run method available)"

            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000

            # Update metrics
            self._capabilities[agent_id]["total_executions"] += 1
            self._capabilities[agent_id]["successful_executions"] += 1
            self._capabilities[agent_id]["total_time_ms"] += execution_time

            return {
                "success": True,
                "agent_id": agent_id,
                "result": result,
                "execution_time_ms": round(execution_time, 2)
            }

        except Exception as e:
            # Update metrics
            self._capabilities[agent_id]["total_executions"] += 1

            return {
                "success": False,
                "agent_id": agent_id,
                "error": str(e)
            }

    async def _tool_run_parallel(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for run_parallel"""
        tasks = args.get("tasks")

        result = await self.run_parallel(tasks)
        return result

    async def run_parallel(self, tasks: List[Union[str, Dict[str, str]]]) -> Dict[str, Any]:
        """
        Execute multiple tasks in parallel with auto-routing support.

        Args:
            tasks: List of tasks - each can be:
                   - String: task description (auto-routed)
                   - Dict: {"task": "...", "agent_id": "..."} (explicit or auto-routed)

        Returns:
            Results from all tasks
        """
        import asyncio

        # Normalize all tasks (auto-route if needed)
        normalized_tasks = await asyncio.gather(*[self._normalize_task(t) for t in tasks])

        async def execute_task(task_info):
            agent_id = task_info["agent_id"]
            task = task_info["task"]

            if agent_id not in self._agents:
                return {"success": False, "error": f"Agent not found: {agent_id}", "task": task}

            agent = self._agents[agent_id]

            try:
                if hasattr(agent, 'run'):
                    result = await agent.run(task)
                else:
                    result = "Agent executed"

                return {"success": True, "agent_id": agent_id, "result": result, "task": task}
            except Exception as e:
                return {"success": False, "agent_id": agent_id, "error": str(e), "task": task}

        # Execute all tasks concurrently
        results = await asyncio.gather(*[execute_task(t) for t in normalized_tasks])

        successful = sum(1 for r in results if r.get("success"))

        return {
            "success": True,
            "total_tasks": len(tasks),
            "successful_tasks": successful,
            "results": results
        }

    async def _tool_run_sequential(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for run_sequential"""
        tasks = args.get("tasks")

        result = await self.run_sequential(tasks)
        return result

    async def run_sequential(self, tasks: List[Union[str, Dict[str, str]]]) -> Dict[str, Any]:
        """
        Execute tasks in sequence with auto-routing support.

        Args:
            tasks: List of tasks - each can be:
                   - String: task description (auto-routed)
                   - Dict: {"task": "...", "agent_id": "..."} (explicit or auto-routed)

        Returns:
            Results from all tasks in order
        """
        results = []
        previous_result = None

        for task_input in tasks:
            # Normalize task (auto-route if needed)
            task_info = await self._normalize_task(task_input)
            agent_id = task_info["agent_id"]
            task = task_info["task"]

            # Include previous result in task context
            if previous_result:
                task = f"{task}\n\nContext from previous step: {previous_result}"

            if agent_id not in self._agents:
                results.append({"success": False, "error": f"Agent not found: {agent_id}", "task": task})
                break

            agent = self._agents[agent_id]

            try:
                if hasattr(agent, 'run'):
                    result = await agent.run(task)
                else:
                    result = "Agent executed"

                results.append({"success": True, "agent_id": agent_id, "result": result, "task": task})
                previous_result = result

            except Exception as e:
                results.append({"success": False, "agent_id": agent_id, "error": str(e), "task": task})
                break

        successful = sum(1 for r in results if r.get("success"))

        return {
            "success": True,
            "total_tasks": len(tasks),
            "completed_tasks": len(results),
            "successful_tasks": successful,
            "results": results,
            "final_result": previous_result if results else None
        }

    async def _tool_create_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for create_workflow"""
        name = args.get("name")
        steps = args.get("steps")

        result = await self.create_workflow(name, steps)
        return result

    async def create_workflow(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a workflow DAG.

        Args:
            name: Workflow name
            steps: List of workflow steps

        Returns:
            Created workflow ID
        """
        workflow = {
            "name": name,
            "steps": steps,
            "created_at": datetime.utcnow().isoformat()
        }

        self._workflows[name] = workflow

        return {
            "success": True,
            "workflow_id": name,
            "steps_count": len(steps)
        }

    async def _tool_run_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for run_workflow"""
        workflow_id = args.get("workflow_id")
        input_data = args.get("input_data", {})

        result = await self.run_workflow(workflow_id, input_data)
        return result

    async def run_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow DAG.

        Args:
            workflow_id: Workflow ID
            input_data: Input data for workflow

        Returns:
            Workflow execution results
        """
        if workflow_id not in self._workflows:
            return {
                "success": False,
                "error": f"Workflow not found: {workflow_id}"
            }

        workflow = self._workflows[workflow_id]
        steps = workflow["steps"]

        # Build dependency graph
        step_results = {}
        completed_steps = set()

        # Execute steps in dependency order
        while len(completed_steps) < len(steps):
            executed_any = False

            for step in steps:
                step_id = step["step_id"]

                if step_id in completed_steps:
                    continue

                # Check if all dependencies are met
                depends_on = step.get("depends_on", [])
                if all(dep in completed_steps for dep in depends_on):
                    # Execute step with auto-routing support
                    task = step["task"]

                    # Auto-route if agent_id not specified
                    if "agent_id" in step:
                        agent_id = step["agent_id"]
                    else:
                        # Auto-route based on task description
                        agent_id = await self._route_task_via_llm(task)

                    # Add dependency results to task context
                    if depends_on:
                        context = "\n".join([
                            f"Result from {dep}: {step_results.get(dep)}"
                            for dep in depends_on
                        ])
                        task = f"{task}\n\nContext:\n{context}"

                    if agent_id not in self._agents:
                        return {
                            "success": False,
                            "error": f"Agent not found for step {step_id}: {agent_id}"
                        }

                    agent = self._agents[agent_id]

                    try:
                        if hasattr(agent, 'run'):
                            result = await agent.run(task)
                        else:
                            result = "Agent executed"

                        step_results[step_id] = result
                        completed_steps.add(step_id)
                        executed_any = True

                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Step {step_id} failed: {str(e)}",
                            "completed_steps": list(completed_steps),
                            "step_results": step_results
                        }

            if not executed_any:
                return {
                    "success": False,
                    "error": "Workflow has circular dependencies or unreachable steps",
                    "completed_steps": list(completed_steps)
                }

        return {
            "success": True,
            "workflow_id": workflow_id,
            "total_steps": len(steps),
            "completed_steps": len(completed_steps),
            "step_results": step_results
        }


def orchestrator(**kwargs) -> OrchestratorPlugin:
    """Create OrchestratorPlugin with simplified interface."""
    return OrchestratorPlugin(**kwargs)
