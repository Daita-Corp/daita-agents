"""
Slack plugin for Daita Agents.

Simple Slack messaging and collaboration - no over-engineering.
"""
import logging
import os
import json
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

class SlackPlugin:
    """
    Simple Slack plugin for agents.
    
    Handles Slack messaging, thread management, and file sharing with agent-specific features.
    """
    
    def __init__(
        self,
        token: str,
        bot_user_oauth_token: Optional[str] = None,
        app_token: Optional[str] = None,
        default_channel: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Slack connection.
        
        Args:
            token: Slack bot token (xoxb-...)
            bot_user_oauth_token: Bot user OAuth token (optional, for extended permissions)
            app_token: App-level token (optional, for Socket Mode)
            default_channel: Default channel for messages (optional)
            **kwargs: Additional Slack client parameters
        """
        if not token or not token.strip():
            raise ValueError("Slack token cannot be empty")
        
        if not token.startswith(('xoxb-', 'xoxp-')):
            raise ValueError("Invalid Slack token format. Expected bot token (xoxb-) or user token (xoxp-)")
        
        self.token = token
        self.bot_user_oauth_token = bot_user_oauth_token
        self.app_token = app_token
        self.default_channel = default_channel
        
        # Store additional config
        self.config = kwargs
        
        self._client = None
        self._user_info = None
        self._channels_cache = {}
        
        logger.debug(f"Slack plugin configured with token: {token[:12]}...")
    
    async def connect(self):
        """Initialize Slack client and validate connection."""
        if self._client is not None:
            return  # Already connected
        
        try:
            from slack_sdk.web.async_client import AsyncWebClient
            from slack_sdk.errors import SlackApiError
            
            # Create Slack client
            self._client = AsyncWebClient(token=self.token)
            
            # Test connection and get bot info
            try:
                auth_response = await self._client.auth_test()
                self._user_info = {
                    'user_id': auth_response.get('user_id'),
                    'team_id': auth_response.get('team_id'),
                    'team': auth_response.get('team'),
                    'user': auth_response.get('user'),
                    'bot_id': auth_response.get('bot_id')
                }
                
                logger.info(f"Connected to Slack as {self._user_info['user']} on team {self._user_info['team']}")
                
            except SlackApiError as e:
                if e.response['error'] == 'invalid_auth':
                    raise RuntimeError("Invalid Slack token. Please check your bot token.")
                elif e.response['error'] == 'account_inactive':
                    raise RuntimeError("Slack account is inactive.")
                else:
                    raise RuntimeError(f"Slack authentication failed: {e.response['error']}")
                    
        except ImportError:
            raise RuntimeError("slack-sdk not installed. Run: pip install slack-sdk")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Slack: {e}")
    
    async def disconnect(self):
        """Close Slack connection."""
        if self._client:
            # Slack SDK client doesn't need explicit closing
            self._client = None
            self._user_info = None
            self._channels_cache = {}
            logger.info("Disconnected from Slack")
    
    async def send_message(
        self,
        channel: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        reply_broadcast: bool = False,
        **kwargs
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
                'channel': channel,
                'text': text,
                'thread_ts': thread_ts,
                'reply_broadcast': reply_broadcast,
                **kwargs
            }
            
            # Add blocks if provided
            if blocks:
                message_args['blocks'] = blocks
            
            # Add attachments if provided (legacy support)
            if attachments:
                message_args['attachments'] = attachments
            
            # Send message
            response = await self._client.chat_postMessage(**message_args)
            
            result = {
                'ok': response['ok'],
                'ts': response['ts'],
                'channel': response['channel'],
                'message': response.get('message', {}),
                'thread_ts': thread_ts
            }
            
            logger.info(f"Sent message to {channel}: {text[:50] if text else 'blocks'}...")
            return result
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            raise RuntimeError(f"Slack send_message failed: {e}")
    
    async def send_agent_summary(
        self,
        channel: str,
        agent_results: Dict[str, Any],
        title: Optional[str] = None,
        thread_ts: Optional[str] = None
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
            blocks = self._format_agent_results(agent_results, title)
            
            # Send message with blocks
            return await self.send_message(
                channel=channel,
                text=f"Agent Summary: {title or 'Results'}", 
                blocks=blocks,
                thread_ts=thread_ts
            )
            
        except Exception as e:
            logger.error(f"Failed to send agent summary: {e}")
            raise RuntimeError(f"Slack send_agent_summary failed: {e}")
    
    async def create_thread_from_workflow(
        self,
        channel: str,
        workflow_results: Dict[str, Any],
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a threaded discussion from workflow results.
        
        Args:
            channel: Channel ID or name
            workflow_results: Workflow execution results
            title: Optional title for the thread
            
        Returns:
            Thread creation response with thread_ts
            
        Example:
            result = await slack.create_thread_from_workflow("#workflows", workflow_results)
        """
        if self._client is None:
            await self.connect()
        
        try:
            # Create initial thread message
            thread_title = title or f"Workflow Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create main thread message
            main_blocks = self._format_workflow_summary(workflow_results, thread_title)
            
            main_response = await self.send_message(
                channel=channel,
                text=f"Workflow Thread: {thread_title}",
                blocks=main_blocks
            )
            
            thread_ts = main_response['ts']
            
            # Add individual agent results as thread replies
            agents = workflow_results.get('agents', {})
            for agent_id, agent_result in agents.items():
                agent_blocks = self._format_agent_results(agent_result, f"Agent: {agent_id}")
                
                await self.send_message(
                    channel=channel,
                    text=f"Agent {agent_id} Results",
                    blocks=agent_blocks,
                    thread_ts=thread_ts
                )
            
            result = {
                'ok': True,
                'thread_ts': thread_ts,
                'channel': channel,
                'agent_count': len(agents),
                'main_message': main_response
            }
            
            logger.info(f"Created workflow thread in {channel} with {len(agents)} agents")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create workflow thread: {e}")
            raise RuntimeError(f"Slack create_thread_from_workflow failed: {e}")
    
    async def upload_file(
        self,
        channel: str,
        file_path: str,
        title: Optional[str] = None,
        initial_comment: Optional[str] = None,
        thread_ts: Optional[str] = None
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
            response = await self._client.files_upload_v2(
                channel=channel,
                file=file_path,
                title=file_title,
                initial_comment=initial_comment,
                thread_ts=thread_ts
            )
            
            result = {
                'ok': response['ok'],
                'file': response.get('file', {}),
                'file_id': response.get('file', {}).get('id'),
                'file_name': file_name,
                'file_title': file_title,
                'channel': channel,
                'thread_ts': thread_ts
            }
            
            logger.info(f"Uploaded file {file_name} to {channel}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload file to Slack: {e}")
            raise RuntimeError(f"Slack upload_file failed: {e}")
    
    async def get_channel_history(
        self,
        channel: str,
        limit: int = 100,
        cursor: Optional[str] = None,
        oldest: Optional[str] = None,
        latest: Optional[str] = None
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
            response = await self._client.conversations_history(
                channel=channel,
                limit=limit,
                cursor=cursor,
                oldest=oldest,
                latest=latest
            )
            
            messages = response.get('messages', [])
            
            # Format messages for easier processing
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    'ts': msg.get('ts'),
                    'user': msg.get('user'),
                    'text': msg.get('text', ''),
                    'type': msg.get('type'),
                    'subtype': msg.get('subtype'),
                    'thread_ts': msg.get('thread_ts'),
                    'reply_count': msg.get('reply_count', 0),
                    'blocks': msg.get('blocks', []),
                    'attachments': msg.get('attachments', [])
                }
                formatted_messages.append(formatted_msg)
            
            logger.info(f"Retrieved {len(formatted_messages)} messages from {channel}")
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Failed to get channel history: {e}")
            raise RuntimeError(f"Slack get_channel_history failed: {e}")
    
    async def get_channels(self, types: str = "public_channel,private_channel") -> List[Dict[str, Any]]:
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
            response = await self._client.conversations_list(types=types)
            
            channels = response.get('channels', [])
            
            # Format channel info
            formatted_channels = []
            for channel in channels:
                formatted_channel = {
                    'id': channel.get('id'),
                    'name': channel.get('name'),
                    'is_channel': channel.get('is_channel'),
                    'is_private': channel.get('is_private'),
                    'is_archived': channel.get('is_archived'),
                    'num_members': channel.get('num_members'),
                    'topic': channel.get('topic', {}).get('value', ''),
                    'purpose': channel.get('purpose', {}).get('value', '')
                }
                formatted_channels.append(formatted_channel)
            
            # Update cache
            self._channels_cache = {ch['name']: ch['id'] for ch in formatted_channels}
            
            logger.info(f"Retrieved {len(formatted_channels)} channels")
            return formatted_channels
            
        except Exception as e:
            logger.error(f"Failed to get channels: {e}")
            raise RuntimeError(f"Slack get_channels failed: {e}")
    
    def _format_agent_results(self, agent_results: Dict[str, Any], title: str) -> List[Dict[str, Any]]:
        """Format agent results as Slack Block Kit blocks."""
        blocks = []
        
        # Header block
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title
            }
        })
        
        # Agent status and timing
        status = agent_results.get('status', 'unknown')
        start_time = agent_results.get('start_time', 'N/A')
        end_time = agent_results.get('end_time', 'N/A')
        duration = agent_results.get('duration_ms', 0)
        
        status_emoji = "" if status == "success" else "" if status == "error" else ""
        
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Status:* {status_emoji} {status.title()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Duration:* {duration:.1f}ms"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Started:* {start_time}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Completed:* {end_time}"
                }
            ]
        })
        
        # Results summary
        if 'output' in agent_results:
            output = agent_results['output']
            if isinstance(output, dict):
                output_text = json.dumps(output, indent=2)
            else:
                output_text = str(output)
            
            # Truncate if too long
            if len(output_text) > 2000:
                output_text = output_text[:2000] + "..."
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Results:*\n```\n{output_text}\n```"
                }
            })
        
        # Error information
        if status == "error" and 'error' in agent_results:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:*\n```\n{agent_results['error']}\n```"
                }
            })
        
        return blocks
    
    def _format_workflow_summary(self, workflow_results: Dict[str, Any], title: str) -> List[Dict[str, Any]]:
        """Format workflow results as Slack Block Kit blocks."""
        blocks = []
        
        # Header block
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title
            }
        })
        
        # Workflow summary
        total_agents = len(workflow_results.get('agents', {}))
        successful_agents = sum(1 for agent in workflow_results.get('agents', {}).values() 
                               if agent.get('status') == 'success')
        failed_agents = total_agents - successful_agents
        
        workflow_status = " Success" if failed_agents == 0 else f" {failed_agents} Failed"
        
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Status:* {workflow_status}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Total Agents:* {total_agents}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Successful:* {successful_agents}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Failed:* {failed_agents}"
                }
            ]
        })
        
        # Add divider
        blocks.append({"type": "divider"})
        
        # Agent summary
        if total_agents > 0:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Agent Results:*\nSee thread replies for detailed results from each agent."
                }
            })
        
        return blocks

    def get_tools(self) -> List['AgentTool']:
        """
        Expose Slack operations as agent tools.

        Returns:
            List of AgentTool instances for Slack messaging operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="send_slack_message",
                description="Send a message to a Slack channel. Use for notifications, alerts, or sharing information with team channels.",
                parameters={
                    "channel": {
                        "type": "string",
                        "description": "Channel name (e.g., #general, #alerts) or channel ID",
                        "required": True
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text to send",
                        "required": True
                    }
                },
                handler=self._tool_send_message,
                category="communication",
                source="plugin",
                plugin_name="Slack",
                timeout_seconds=30
            ),
            AgentTool(
                name="send_slack_summary",
                description="Send a formatted agent summary to Slack with results and metadata. Use for reporting agent execution results.",
                parameters={
                    "channel": {
                        "type": "string",
                        "description": "Channel name or ID to send summary to",
                        "required": True
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary text describing the results",
                        "required": True
                    },
                    "results": {
                        "type": "object",
                        "description": "Results data to include in the summary",
                        "required": False
                    }
                },
                handler=self._tool_send_summary,
                category="communication",
                source="plugin",
                plugin_name="Slack",
                timeout_seconds=30
            ),
            AgentTool(
                name="list_slack_channels",
                description="List all Slack channels the bot has access to",
                parameters={},
                handler=self._tool_list_channels,
                category="communication",
                source="plugin",
                plugin_name="Slack",
                timeout_seconds=30
            )
        ]

    async def _tool_send_message(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for send_slack_message"""
        channel = args.get("channel")
        text = args.get("text")

        result = await self.send_message(channel, text)

        return {
            "success": True,
            "channel": result.get("channel"),
            "timestamp": result.get("ts"),
            "message_sent": True
        }

    async def _tool_send_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for send_slack_summary"""
        channel = args.get("channel")
        summary = args.get("summary")
        results = args.get("results", {})

        await self.send_agent_summary(
            channel=channel,
            agent_results={"summary": summary, "data": results}
        )

        return {
            "success": True,
            "channel": channel,
            "summary_sent": True
        }

    async def _tool_list_channels(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_slack_channels"""
        channels = await self.get_channels()

        # Simplify channel data for LLM
        simplified = [
            {
                "name": ch["name"],
                "id": ch["id"],
                "is_private": ch["is_private"],
                "members": ch.get("num_members", 0)
            }
            for ch in channels
        ]

        return {
            "success": True,
            "channels": simplified,
            "count": len(simplified)
        }

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def slack(**kwargs) -> SlackPlugin:
    """Create Slack plugin with simplified interface."""
    return SlackPlugin(**kwargs)