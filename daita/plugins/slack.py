"""
Slack plugin for Daita Agents.

Simple Slack messaging and collaboration - no over-engineering.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, TypedDict

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    ToolView,
)

from .base import ConnectorPlugin
from .manifest import PluginKind, PluginManifest
from ..core.exceptions import AuthenticationError, PluginError

if TYPE_CHECKING:
    from slack_sdk.web.async_client import AsyncWebClient

    from ..core.tools import LocalTool

logger = logging.getLogger(__name__)


_SLACK_TOOL_DEFINITIONS = (
    {
        "name": "send_slack_message",
        "capability_id": "slack.message.send",
        "operation_type": "slack.message.send",
        "description": "Send a message to a Slack channel or user.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel name (e.g., #general) or channel ID",
                },
                "text": {
                    "type": "string",
                    "description": "Message text to send",
                },
            },
            "required": ["channel", "text"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_send_message",
    },
    {
        "name": "send_slack_summary",
        "capability_id": "slack.summary.send",
        "operation_type": "slack.summary.send",
        "description": "Send a formatted agent results summary to a Slack channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel name or ID",
                },
                "summary": {
                    "type": "string",
                    "description": "Summary text describing the results",
                },
                "results": {
                    "type": "object",
                    "description": "Optional results data to include",
                },
            },
            "required": ["channel", "summary"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_send_summary",
    },
    {
        "name": "list_slack_channels",
        "capability_id": "slack.channel.list",
        "operation_type": "slack.channel.list",
        "description": "List all Slack channels the bot has access to.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_list_channels",
    },
    {
        "name": "read_slack_messages",
        "capability_id": "slack.message.read",
        "operation_type": "slack.message.read",
        "description": "Read recent messages from a Slack channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel ID or name (e.g., #general)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return (default: 20)",
                },
            },
            "required": ["channel"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_read_messages",
    },
)


class _SlackUserInfo(TypedDict):
    user_id: Optional[str]
    team_id: Optional[str]
    team: Optional[str]
    user: Optional[str]
    bot_id: Optional[str]


class _SlackExecutor:
    """Execute Slack runtime capabilities and return typed evidence."""

    id = "slack.operations"
    capability_ids = frozenset(
        definition["capability_id"] for definition in _SLACK_TOOL_DEFINITIONS
    )

    def __init__(self, plugin: "SlackPlugin") -> None:
        self._plugin = plugin

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        tool_view_name = (
            context.get("tool_view", {}).get("name")
            if isinstance(context, dict)
            else None
        ) or definition["name"]
        return [
            Evidence(
                kind="slack.operation.result",
                owner=self._plugin.manifest.id,
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "operation": definition["name"],
                    "request": dict(task.input or {}),
                    "result": result,
                },
                metadata={
                    "capability_id": task.capability_id,
                    "tool_view": tool_view_name,
                },
            )
        ]


class SlackPlugin(ConnectorPlugin):
    """
    Simple Slack plugin for agents.

    Handles Slack messaging, thread management, and file sharing with agent-specific features.
    """

    manifest = PluginManifest(
        id="slack",
        display_name="Slack",
        version="2.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"slack", "messaging", "collaboration"}),
        provides=frozenset({"slack_read", "slack_send"}),
    )

    def __init__(self, token: str, default_channel: Optional[str] = None, **kwargs):
        """
        Initialize Slack connection.

        Args:
            token: Slack bot token (xoxb-...)
            default_channel: Default channel for messages (optional)
            **kwargs: Additional Slack client parameters
        """
        if not token or not token.strip():
            raise ValueError("Slack token cannot be empty")

        if not token.startswith(("xoxb-", "xoxp-")):
            raise ValueError(
                "Invalid Slack token format. Expected bot token (xoxb-) or user token (xoxp-)"
            )

        self.token = token
        self.default_channel = default_channel

        # Store additional config
        self.config = kwargs

        self._client: Optional["AsyncWebClient"] = None
        self._user_info: Optional[_SlackUserInfo] = None
        self._executor = _SlackExecutor(self)
        self._connect_lock = asyncio.Lock()

        logger.debug(f"Slack plugin configured with token: {token[:12]}...")

    @property
    def is_connected(self) -> bool:
        """Whether the Slack client has been initialized."""
        return self._client is not None

    @property
    def client(self) -> "AsyncWebClient":
        """Return the authenticated Slack client."""
        if self._client is None:
            raise PluginError("Slack is not connected", plugin_name="Slack")
        return self._client

    async def teardown(self) -> None:
        """Release runtime-owned Slack resources."""
        await self.disconnect()

    def declare_capabilities(self) -> tuple[Capability, ...]:
        """Declare Slack operations as runtime-plannable capabilities."""
        return tuple(
            Capability(
                id=definition["capability_id"],
                owner=self.manifest.id,
                description=definition["description"],
                domains=frozenset({"slack", "messaging", "collaboration"}),
                operation_types=frozenset({definition["operation_type"]}),
                access=definition["access"],
                risk=definition["risk"],
                input_schema=definition["parameters"],
                output_evidence=frozenset({"slack.operation.result"}),
                executor=self._executor.id,
                model_visible=True,
                retry_safe=definition["retry_safe"],
                replay_safe=definition["retry_safe"],
                idempotent=definition["idempotent"],
                side_effecting=definition["side_effecting"],
                timeout_seconds=30,
                metadata={"tool_name": definition["name"]},
            )
            for definition in _SLACK_TOOL_DEFINITIONS
        )

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        """Declare the typed evidence returned by Slack capability execution."""
        return (
            EvidenceSchema(
                kind="slack.operation.result",
                owner=self.manifest.id,
                description="Slack operation result evidence.",
                json_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "request": {"type": "object"},
                        "result": {"type": "object"},
                    },
                    "required": ["operation", "request", "result"],
                },
            ),
        )

    def get_executors(self) -> tuple[_SlackExecutor, ...]:
        """Return the executor for Slack runtime capabilities."""
        return (self._executor,)

    def get_tool_views(self) -> tuple[ToolView, ...]:
        """Expose Slack capabilities as model-visible tool views."""
        return tuple(
            ToolView(
                name=definition["name"],
                capability_id=definition["capability_id"],
                description=definition["description"],
                parameters=definition["parameters"],
            )
            for definition in _SLACK_TOOL_DEFINITIONS
        )

    def _definition_for_capability(self, capability_id: str) -> dict[str, Any]:
        for definition in _SLACK_TOOL_DEFINITIONS:
            if definition["capability_id"] == capability_id:
                return definition
        raise KeyError(capability_id)

    async def connect(self):
        """Initialize Slack client and validate connection."""
        async with self._connect_lock:
            if self._client is not None:
                return

            try:
                from slack_sdk.errors import SlackApiError
                from slack_sdk.web.async_client import AsyncWebClient

                client = AsyncWebClient(token=self.token)
                try:
                    auth_response = await client.auth_test()
                except SlackApiError as error:
                    error_code = error.response.get("error", "unknown_error")
                    if error_code == "invalid_auth":
                        message = "Invalid Slack token. Please check your bot token."
                    elif error_code == "account_inactive":
                        message = "Slack account is inactive."
                    else:
                        message = f"Slack authentication failed: {error_code}"
                    raise AuthenticationError(message, provider="Slack") from error

                user_info: _SlackUserInfo = {
                    "user_id": auth_response.get("user_id"),
                    "team_id": auth_response.get("team_id"),
                    "team": auth_response.get("team"),
                    "user": auth_response.get("user"),
                    "bot_id": auth_response.get("bot_id"),
                }
                self._client = client
                self._user_info = user_info

                logger.info(
                    "Connected to Slack as %s on team %s",
                    user_info["user"],
                    user_info["team"],
                )

            except ImportError as error:
                raise ImportError(
                    "slack-sdk is required. Install with: "
                    "pip install 'daita-agents[slack]'"
                ) from error
            except (AuthenticationError, PluginError):
                raise
            except Exception as error:
                raise PluginError(
                    f"Failed to connect to Slack: {error}", plugin_name="Slack"
                ) from error

    async def disconnect(self):
        """Close Slack connection."""
        if self._client:
            # Slack SDK client doesn't need explicit closing
            self._client = None
            self._user_info = None
            logger.info("Disconnected from Slack")

    async def send_message(
        self,
        channel: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        reply_broadcast: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a message to a Slack channel.

        Args:
            channel: Channel ID or name (#channel, @user)
            text: Message text (required if no blocks)
            blocks: Slack Block Kit blocks for rich formatting
            attachments: Message attachments (legacy, use blocks instead)
            thread_ts: Timestamp of parent message (for threaded replies)
            reply_broadcast: Whether to broadcast thread reply to channel
            **kwargs: Additional message parameters

        Returns:
            Message response with timestamp and metadata

        Example:
            result = await slack.send_message("#alerts", "System update complete")
        """
        if self._client is None:
            await self.connect()

        # Use default channel if not specified
        if not channel and self.default_channel:
            channel = self.default_channel

        if not channel:
            raise ValueError("Channel must be specified or default_channel must be set")

        # Validate message content
        if not text and not blocks:
            raise ValueError("Either text or blocks must be provided")

        try:
            # Prepare message arguments
            message_args = {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
                "reply_broadcast": reply_broadcast,
                **kwargs,
            }

            # Add blocks if provided
            if blocks:
                message_args["blocks"] = blocks

            # Add attachments if provided (legacy support)
            if attachments:
                message_args["attachments"] = attachments

            # Send message
            response = await self.client.chat_postMessage(**message_args)

            result = {
                "ok": response["ok"],
                "ts": response["ts"],
                "channel": response["channel"],
                "message": response.get("message", {}),
                "thread_ts": thread_ts,
            }

            logger.info(
                f"Sent message to {channel}: {text[:50] if text else 'blocks'}..."
            )
            return result

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            raise PluginError(f"Slack send_message failed: {e}", plugin_name="Slack")

    async def send_agent_summary(
        self,
        channel: str,
        agent_results: Dict[str, Any],
        title: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a formatted summary of agent results to Slack.

        Args:
            channel: Channel ID or name
            agent_results: Agent execution results
            title: Optional title for the summary
            thread_ts: Optional thread timestamp

        Returns:
            Message response

        Example:
            result = await slack.send_agent_summary("#data-team", agent_results)
        """
        if self._client is None:
            await self.connect()

        try:
            # Create formatted blocks for agent results
            blocks = self._format_agent_results(agent_results, title or "Agent Results")

            # Send message with blocks
            return await self.send_message(
                channel=channel,
                text=f"Agent Summary: {title or 'Results'}",
                blocks=blocks,
                thread_ts=thread_ts,
            )

        except Exception as e:
            logger.error(f"Failed to send agent summary: {e}")
            raise PluginError(
                f"Slack send_agent_summary failed: {e}", plugin_name="Slack"
            )

    async def upload_file(
        self,
        channel: str,
        file_path: str,
        title: Optional[str] = None,
        initial_comment: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to Slack.

        Args:
            channel: Channel ID or name
            file_path: Path to file to upload
            title: File title (defaults to filename)
            initial_comment: Comment to add with file
            thread_ts: Thread timestamp (for threaded uploads)

        Returns:
            File upload response

        Example:
            result = await slack.upload_file("#reports", "analysis.pdf", "Monthly Analysis")
        """
        if self._client is None:
            await self.connect()

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Get file info
            file_name = os.path.basename(file_path)
            file_title = title or file_name

            # Upload file
            response = await self.client.files_upload_v2(
                channel=channel,
                file=file_path,
                title=file_title,
                initial_comment=initial_comment,
                thread_ts=thread_ts,
            )

            result = {
                "ok": response["ok"],
                "file": response.get("file", {}),
                "file_id": response.get("file", {}).get("id"),
                "file_name": file_name,
                "file_title": file_title,
                "channel": channel,
                "thread_ts": thread_ts,
            }

            logger.info(f"Uploaded file {file_name} to {channel}")
            return result

        except Exception as e:
            logger.error(f"Failed to upload file to Slack: {e}")
            raise PluginError(f"Slack upload_file failed: {e}", plugin_name="Slack")

    async def get_channel_history(
        self,
        channel: str,
        limit: int = 100,
        cursor: Optional[str] = None,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get channel message history.

        Args:
            channel: Channel ID or name
            limit: Maximum number of messages to return
            cursor: Pagination cursor
            oldest: Oldest timestamp to include
            latest: Latest timestamp to include

        Returns:
            List of messages

        Example:
            messages = await slack.get_channel_history("#alerts", limit=50)
        """
        if self._client is None:
            await self.connect()

        try:
            # Get conversation history
            response = await self.client.conversations_history(
                channel=channel,
                limit=limit,
                cursor=cursor,
                oldest=oldest,
                latest=latest,
            )

            messages = response.get("messages", [])

            # Format messages for easier processing
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "ts": msg.get("ts"),
                    "user": msg.get("user"),
                    "text": msg.get("text", ""),
                    "type": msg.get("type"),
                    "subtype": msg.get("subtype"),
                    "thread_ts": msg.get("thread_ts"),
                    "reply_count": msg.get("reply_count", 0),
                    "blocks": msg.get("blocks", []),
                    "attachments": msg.get("attachments", []),
                }
                formatted_messages.append(formatted_msg)

            logger.info(f"Retrieved {len(formatted_messages)} messages from {channel}")
            return formatted_messages

        except Exception as e:
            logger.error(f"Failed to get channel history: {e}")
            raise PluginError(
                f"Slack get_channel_history failed: {e}", plugin_name="Slack"
            )

    async def get_channels(
        self, types: str = "public_channel,private_channel"
    ) -> List[Dict[str, Any]]:
        """
        Get list of channels the bot has access to.

        Args:
            types: Channel types to include (comma-separated)

        Returns:
            List of channel information

        Example:
            channels = await slack.get_channels()
        """
        if self._client is None:
            await self.connect()

        try:
            # Get conversations list
            response = await self.client.conversations_list(types=types)

            channels = response.get("channels", [])

            # Format channel info
            formatted_channels = []
            for channel in channels:
                formatted_channel = {
                    "id": channel.get("id"),
                    "name": channel.get("name"),
                    "is_channel": channel.get("is_channel"),
                    "is_private": channel.get("is_private"),
                    "is_archived": channel.get("is_archived"),
                    "num_members": channel.get("num_members"),
                    "topic": channel.get("topic", {}).get("value", ""),
                    "purpose": channel.get("purpose", {}).get("value", ""),
                }
                formatted_channels.append(formatted_channel)

            logger.info(f"Retrieved {len(formatted_channels)} channels")
            return formatted_channels

        except Exception as e:
            logger.error(f"Failed to get channels: {e}")
            raise PluginError(f"Slack get_channels failed: {e}", plugin_name="Slack")

    def _format_agent_results(
        self, agent_results: Dict[str, Any], title: str
    ) -> List[Dict[str, Any]]:
        """Format agent results as Slack Block Kit blocks."""
        blocks: List[Dict[str, Any]] = []

        # Header block
        blocks.append({"type": "header", "text": {"type": "plain_text", "text": title}})

        # Agent status and timing
        status = agent_results.get("status", "unknown")
        start_time = agent_results.get("start_time", "N/A")
        end_time = agent_results.get("end_time", "N/A")
        duration = agent_results.get("duration_ms", 0)

        status_emoji = (
            "[ok]" if status == "success" else "[err]" if status == "error" else "[?]"
        )

        blocks.append(
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:* {status_emoji} {status.title()}",
                    },
                    {"type": "mrkdwn", "text": f"*Duration:* {duration:.1f}ms"},
                    {"type": "mrkdwn", "text": f"*Started:* {start_time}"},
                    {"type": "mrkdwn", "text": f"*Completed:* {end_time}"},
                ],
            }
        )

        # Results summary
        if "output" in agent_results:
            output = agent_results["output"]
            if isinstance(output, dict):
                output_text = json.dumps(output, indent=2)
            else:
                output_text = str(output)

            # Truncate if too long
            if len(output_text) > 2000:
                output_text = output_text[:2000] + "..."

            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Results:*\n```\n{output_text}\n```",
                    },
                }
            )

        # Error information
        if status == "error" and "error" in agent_results:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Error:*\n```\n{agent_results['error']}\n```",
                    },
                }
            )

        return blocks

    async def _tool_send_message(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for send_slack_message"""
        channel = args.get("channel")
        text = args.get("text")

        if not isinstance(channel, str) or not channel:
            raise PluginError(
                "send_slack_message requires a non-empty channel",
                plugin_name="Slack",
            )
        if not isinstance(text, str) or not text:
            raise PluginError(
                "send_slack_message requires non-empty text", plugin_name="Slack"
            )

        result = await self.send_message(channel, text)

        return {
            "channel": result.get("channel"),
            "timestamp": result.get("ts"),
            "message_sent": True,
        }

    async def _tool_send_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for send_slack_summary"""
        channel = args.get("channel")
        summary = args.get("summary")
        results = args.get("results", {})

        if not isinstance(channel, str) or not channel:
            raise PluginError(
                "send_slack_summary requires a non-empty channel",
                plugin_name="Slack",
            )
        if not isinstance(summary, str) or not summary:
            raise PluginError(
                "send_slack_summary requires a non-empty summary",
                plugin_name="Slack",
            )
        if not isinstance(results, dict):
            raise PluginError(
                "send_slack_summary results must be an object", plugin_name="Slack"
            )

        await self.send_agent_summary(
            channel=channel, agent_results={"summary": summary, "data": results}
        )

        return {"channel": channel, "summary_sent": True}

    async def _tool_list_channels(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_slack_channels"""
        channels = await self.get_channels()

        # Simplify channel data for LLM
        simplified = [
            {
                "name": ch["name"],
                "id": ch["id"],
                "is_private": ch["is_private"],
                "members": ch.get("num_members", 0),
            }
            for ch in channels
        ]

        return {"channels": simplified, "count": len(simplified)}

    async def _tool_read_messages(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for read_slack_messages"""
        channel = args.get("channel")
        limit = args.get("limit", 20)

        if not isinstance(channel, str) or not channel:
            raise PluginError(
                "read_slack_messages requires a non-empty channel",
                plugin_name="Slack",
            )
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise PluginError(
                "read_slack_messages limit must be a positive integer",
                plugin_name="Slack",
            )

        messages = await self.get_channel_history(channel=channel, limit=limit)

        # Return simplified message data
        simplified = [
            {
                "ts": msg.get("ts"),
                "user": msg.get("user"),
                "text": msg.get("text", ""),
                "thread_ts": msg.get("thread_ts"),
                "reply_count": msg.get("reply_count", 0),
            }
            for msg in messages
        ]

        return {"channel": channel, "messages": simplified, "count": len(simplified)}

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def slack(**kwargs) -> SlackPlugin:
    """Create Slack plugin with simplified interface."""
    return SlackPlugin(**kwargs)
